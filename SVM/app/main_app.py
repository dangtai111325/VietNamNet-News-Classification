"""
Ứng dụng phân loại tin tức Vietnamnet
Chạy: streamlit run main_app.py
"""

import os, re, pickle, datetime, warnings
import requests
import urllib3
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Đường dẫn tự động ─────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.normpath(os.path.join(_HERE, "..", "model", "inference_pipeline.pkl"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Load pipeline (cache — chỉ load 1 lần) ────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)


# ── Scraper ────────────────────────────────────────────────────────────────────
def scrape_article(url: str) -> tuple:
    resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = ""
    for sel in [
        "h1.content-detail-title", "h1.ArticleTitle", "h1.title-detail",
        "h1[class*='title']", "h1",
    ]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            if len(t) > 5:
                title = t
                break

    content = ""
    for sel in [
        "div.ArticleContent", "div.content-detail-body",
        "div[class*='ArticleContent']", "div[class*='article-content']",
        "div[class*='content-detail']", "article", "main",
    ]:
        el = soup.select_one(sel)
        if el:
            paras = [p.get_text(strip=True) for p in el.find_all("p")
                     if len(p.get_text(strip=True)) > 20]
            if paras:
                content = " ".join(paras)
                break

    if not content:
        content = " ".join(
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 30
        )

    return title.strip(), content.strip()


# ── Tiền xử lý (khớp chính xác với pipeline training) ─────────────────────────
def preprocess(title: str, content: str, pipeline: dict) -> str:
    from pyvi import ViTokenizer
    sw   = set(pipeline["stopwords"])
    text = (str(title) + " " + str(title) + " " + str(content)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+",     " ", text)
    text = ViTokenizer.tokenize(text)
    return re.sub(r"\s+", " ",
                  " ".join(t for t in text.split() if t not in sw)).strip()


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(title: str, content: str, pipeline: dict) -> dict:
    if not title.strip() and not content.strip():
        raise ValueError("Không có nội dung để phân loại.")

    clean   = preprocess(title, content, pipeline)
    vec     = pipeline["vectorizer"].transform([clean])
    classes = pipeline["classes"]

    # predict_proba() dung Platt scaling (calibrated) — chinh xac hon decision_function + softmax
    probs   = pipeline["model"].predict_proba(vec)[0]
    top_idx = np.argsort(probs)[::-1]

    return {
        "pred_class":      classes[top_idx[0]],
        "confidence":      float(probs[top_idx[0]]),
        "top5":            [(classes[i], float(probs[i])) for i in top_idx[:5]],
        "title":           title,
        "content_preview": content[:600] + ("…" if len(content) > 600 else ""),
        "n_tokens":        len(clean.split()),
    }


# ── Top keywords từ LinearSVC coef_ ───────────────────────────────────────────
def get_top_keywords(pipeline: dict, pred_class: str, n: int = 10) -> list:
    classes       = pipeline["classes"]
    class_idx     = classes.index(pred_class)
    coef          = pipeline["model"].coef_[class_idx]
    feature_names = pipeline["vectorizer"].get_feature_names_out()
    top_idx       = np.argsort(coef)[::-1][:n]
    return [(feature_names[i], float(coef[i])) for i in top_idx]


# ── Hiển thị kết quả ──────────────────────────────────────────────────────────
def show_result(res: dict, pipeline: dict, source_label: str = ""):
    conf_pct = res["confidence"] * 100
    gap      = (res["top5"][0][1] - res["top5"][1][1]) if len(res["top5"]) > 1 else 1.0

    # Cảnh báo low confidence
    if gap < 0.05:
        st.warning(
            f"⚠️ Độ phân cách thấp — bài này có thể thuộc cả "
            f"**{res['top5'][0][0]}** lẫn **{res['top5'][1][0]}** "
            f"(chênh lệch chỉ {gap*100:.1f}%)"
        )

    if conf_pct >= 60:
        color, badge = "#27ae60", "Tin cậy cao"
    elif conf_pct >= 40:
        color, badge = "#e67e22", "Tin cậy trung bình"
    else:
        color, badge = "#e74c3c", "Tin cậy thấp"

    col_left, col_right = st.columns(2)

    # ── Cột trái ──────────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Kết quả phân loại")
        st.markdown(
            f"""<div style='background:{color}18;border-left:5px solid {color};
                           padding:16px 20px;border-radius:8px;margin-bottom:14px'>
                    <div style='font-size:1.5rem;font-weight:800;color:{color};margin-bottom:4px'>
                        {res["pred_class"]}
                    </div>
                    <div style='color:#555;font-size:.9rem'>
                        Độ tin cậy: <b>{conf_pct:.1f}%</b> &nbsp;·&nbsp; {badge}
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Top 5 chủ đề có thể:**")
        for i, (cls, prob) in enumerate(res["top5"]):
            label = f"🥇 **{cls}**" if i == 0 else f"{i+1}. {cls}"
            st.markdown(f"{label} &nbsp; `{prob*100:.1f}%`", unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown("**Top 10 từ khoá quyết định:**")
        kws    = get_top_keywords(pipeline, res["pred_class"], n=10)
        df_kws = pd.DataFrame(kws, columns=["Từ / Bigram", "Trọng số (coef)"])
        df_kws.index += 1
        st.dataframe(
            df_kws.style.format({"Trọng số (coef)": "{:.4f}"}),
            use_container_width=True,
            height=340,
        )

    # ── Cột phải ──────────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Nội dung bài báo")
        if res["title"]:
            st.markdown(f"**Tiêu đề:** {res['title']}")
        st.caption(f"Tokens sau tiền xử lý: {res['n_tokens']:,}")
        st.text_area(
            "_content",
            value=res["content_preview"] or "(không có nội dung)",
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    # Lưu lịch sử
    st.session_state.history.insert(0, {
        "Thời gian": datetime.datetime.now().strftime("%H:%M:%S"),
        "Nguồn":     source_label[:70] or res["title"][:70] or "—",
        "Chủ đề":    res["pred_class"],
        "Tin cậy":   f"{conf_pct:.1f}%",
    })


# ── Tab Lịch sử ───────────────────────────────────────────────────────────────
def show_history():
    if not st.session_state.get("history"):
        st.info("Chưa có lịch sử phân loại trong session này.")
        return

    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, hide_index=True, use_container_width=True)

    col_dl, col_clr, _ = st.columns([2, 2, 6])
    with col_dl:
        st.download_button(
            "Tải CSV",
            data=df_hist.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clr:
        if st.button("Xoá lịch sử", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Phân loại tin tức Vietnamnet",
        page_icon="🇻🇳",
        layout="wide",
    )

    st.title("🇻🇳 Phân loại tin tức Vietnamnet")
    st.markdown(
        "Phân loại tự động bài báo vào **19 chủ đề** bằng **LinearSVC + TF-IDF**."
    )

    try:
        pipeline = load_pipeline()
    except FileNotFoundError:
        st.error(
            f"❌ Không tìm thấy pipeline:\n\n`{PIPELINE_PATH}`\n\n"
            "Chạy **Section 7** trong `main_SVM.ipynb` để tạo file này."
        )
        st.stop()

    st.caption(
        f"`inference_pipeline.pkl`  |  {len(pipeline['classes'])} chủ đề  |  "
        f"max_features={pipeline['config'].get('max_features', '?'):,}  |  "
        f"C={pipeline['config'].get('C', '?')}"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 Nhập URL", "📝 Nhập text", "📋 Batch URL", "📜 Lịch sử",
    ])

    # ── Tab 1: Nhập URL ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("Nhập URL bài báo rồi nhấn **Enter** hoặc **Phân loại**.")
        with st.form("form_url", border=False):
            url = st.text_input(
                "URL",
                placeholder="https://vietnamnet.vn/...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Phân loại", type="primary")

        if submitted:
            if not url.strip():
                st.warning("Vui lòng nhập URL.")
            else:
                with st.spinner("Đang tải bài báo..."):
                    try:
                        title, content = scrape_article(url.strip())
                    except Exception as e:
                        st.error(f"❌ Không thể tải URL: {e}")
                        st.info("Nếu URL bị chặn, dùng tab **📝 Nhập text** để dán nội dung thủ công.")
                        st.stop()

                if not title and not content:
                    st.error("❌ Không tìm thấy nội dung — thử tab **📝 Nhập text**.")
                else:
                    with st.spinner("Đang phân loại..."):
                        res = predict(title, content, pipeline)
                    st.divider()
                    show_result(res, pipeline, source_label=url.strip())

    # ── Tab 2: Nhập text ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "Dùng khi URL không lấy được nội dung (paywall, Cloudflare, lỗi scrape…)."
        )
        with st.form("form_text", border=False):
            t_title   = st.text_input(
                "Tiêu đề bài báo", placeholder="Nhập hoặc dán tiêu đề…"
            )
            t_content = st.text_area(
                "Nội dung bài báo",
                placeholder="Dán nội dung bài báo vào đây…",
                height=220,
            )
            sub2 = st.form_submit_button("Phân loại", type="primary")

        if sub2:
            if not t_title.strip() and not t_content.strip():
                st.warning("Vui lòng nhập ít nhất tiêu đề hoặc nội dung.")
            else:
                with st.spinner("Đang phân loại..."):
                    res2 = predict(t_title.strip(), t_content.strip(), pipeline)
                st.divider()
                show_result(res2, pipeline,
                            source_label=t_title.strip()[:70] or "text input")

    # ── Tab 3: Batch URL ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("Dán nhiều URL (mỗi dòng 1 URL) — phân loại hàng loạt rồi tải CSV.")
        with st.form("form_batch", border=False):
            batch_raw = st.text_area(
                "Danh sách URL",
                placeholder=(
                    "https://vietnamnet.vn/bai-bao-1\n"
                    "https://vietnamnet.vn/bai-bao-2\n..."
                ),
                height=200,
                label_visibility="collapsed",
            )
            sub3 = st.form_submit_button("Phân loại tất cả", type="primary")

        if sub3:
            urls_list = [u.strip() for u in batch_raw.strip().splitlines() if u.strip()]
            if not urls_list:
                st.warning("Không tìm thấy URL nào.")
            else:
                batch_results = []
                prog = st.progress(0.0, text="Đang xử lý…")

                for idx, u in enumerate(urls_list):
                    prog.progress(
                        (idx + 1) / len(urls_list),
                        text=f"[{idx+1}/{len(urls_list)}] {u[:70]}…",
                    )
                    try:
                        _title, _content = scrape_article(u)
                        _res = predict(_title, _content, pipeline)
                        batch_results.append({
                            "URL":        u,
                            "Tiêu đề":    _title[:80],
                            "Chủ đề":     _res["pred_class"],
                            "Tin cậy":    f"{_res['confidence']*100:.1f}%",
                            "Trạng thái": "✅ OK",
                        })
                        st.session_state.history.insert(0, {
                            "Thời gian": datetime.datetime.now().strftime("%H:%M:%S"),
                            "Nguồn":     u[:70],
                            "Chủ đề":    _res["pred_class"],
                            "Tin cậy":   f"{_res['confidence']*100:.1f}%",
                        })
                    except Exception as e:
                        batch_results.append({
                            "URL":        u,
                            "Tiêu đề":    "—",
                            "Chủ đề":     "—",
                            "Tin cậy":    "—",
                            "Trạng thái": f"❌ {e}",
                        })

                prog.empty()
                df_batch = pd.DataFrame(batch_results)
                ok_count = (df_batch["Trạng thái"] == "✅ OK").sum()
                st.success(f"Hoàn thành: {ok_count}/{len(urls_list)} URL thành công.")
                st.dataframe(df_batch, hide_index=True, use_container_width=True)
                st.download_button(
                    "Tải kết quả (CSV)",
                    data=df_batch.to_csv(
                        index=False, encoding="utf-8-sig"
                    ).encode("utf-8-sig"),
                    file_name="batch_results.csv",
                    mime="text/csv",
                )

    # ── Tab 4: Lịch sử ────────────────────────────────────────────────────────
    with tab4:
        show_history()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("19 chủ đề của mô hình"):
        cols = st.columns(4)
        for i, cls in enumerate(pipeline["classes"]):
            cols[i % 4].markdown(f"- {cls}")


if __name__ == "__main__":
    main()
