"""
Combined App — SVM + PhoBERT Ensemble
Phân loại tin tức Vietnamnet bằng điểm tin cậy kết hợp.

Chạy: streamlit run app_combined.py
"""

import os, re, json, pickle, datetime, warnings
import requests
import urllib3
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
SVM_PIPELINE   = os.path.normpath(os.path.join(_HERE, "..", "SVM",     "model", "inference_pipeline.pkl"))
PHO_MODEL_DIR  = os.path.normpath(os.path.join(_HERE, "..", "PhoBERT", "model"))
PHO_CONFIG     = os.path.join(PHO_MODEL_DIR, "label_config.json")
PHO_THRESHOLD  = os.path.join(PHO_MODEL_DIR, "thresholds.json")
MAX_LENGTH     = 256

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Load models (cache) ───────────────────────────────────────────────────────
@st.cache_resource
def load_svm():
    with open(SVM_PIPELINE, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_phobert():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(PHO_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(PHO_MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(PHO_MODEL_DIR)
    model.to(device)
    model.eval()

    thresholds  = None
    temperature = 1.0
    if os.path.exists(PHO_THRESHOLD):
        try:
            thr         = json.load(open(PHO_THRESHOLD, encoding="utf-8"))
            temperature = float(thr.get("temperature", 1.0))
            thresholds  = np.array(
                [thr["thresholds"].get(cls, 1.0) for cls in cfg["classes"]],
                dtype=np.float32,
            )
        except Exception:
            pass
    return tokenizer, model, cfg, thresholds, temperature, device


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


# ── Tiền xử lý SVM ────────────────────────────────────────────────────────────
def preprocess_svm(title: str, content: str, pipeline: dict) -> str:
    from pyvi import ViTokenizer
    sw   = set(pipeline["stopwords"])
    text = (str(title) + " " + str(title) + " " + str(content)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+",     " ", text)
    text = ViTokenizer.tokenize(text)
    return re.sub(r"\s+", " ", " ".join(t for t in text.split() if t not in sw)).strip()


def score_to_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    exp_s  = np.exp(scores - scores.max())
    denom  = exp_s.sum()
    return exp_s / denom if denom else np.full_like(exp_s, 1.0 / len(exp_s))


# ── Tiền xử lý PhoBERT ────────────────────────────────────────────────────────
def preprocess_phobert(title: str, content: str) -> str:
    from pyvi import ViTokenizer
    text = (str(title) + " " + str(content)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+",     " ", text)
    text = ViTokenizer.tokenize(text)
    return re.sub(r"\s+", " ", text).strip()


# ── Head-Tail tokenize ────────────────────────────────────────────────────────
def head_tail_encode(text: str, tokenizer, device: str) -> dict:
    import torch
    half   = (MAX_LENGTH - 2) // 2
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    if len(tokens) <= MAX_LENGTH - 2:
        enc  = tokenizer(text, truncation=True, padding="max_length",
                         max_length=MAX_LENGTH, return_tensors="pt")
        ids  = enc["input_ids"][0].tolist()
        attn = enc["attention_mask"][0].tolist()
    else:
        head  = tokens[:half]
        tail  = tokens[-half:]
        ids   = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
        attn  = [1] * len(ids)
        pad_n = MAX_LENGTH - len(ids)
        ids  += [tokenizer.pad_token_id] * pad_n
        attn += [0] * pad_n

    return {
        "input_ids":      torch.tensor([ids],  dtype=torch.long).to(device),
        "attention_mask": torch.tensor([attn], dtype=torch.long).to(device),
    }


# ── Inference từng model ──────────────────────────────────────────────────────
def _predict_svm(title: str, content: str, pipeline: dict) -> np.ndarray:
    """Trả về điểm tin cậy suy ra từ decision_function(), theo thứ tự pipeline['classes'].""" 
    clean  = preprocess_svm(title, content, pipeline)
    vec    = pipeline["vectorizer"].transform([clean])
    scores = pipeline["model"].decision_function(vec)[0]
    return score_to_probs(scores)


def _predict_phobert(title: str, content: str,
                     tokenizer, model, cfg: dict,
                     thresholds, temperature: float, device: str) -> np.ndarray:
    """Trả về mảng xác suất (softmax + threshold calibration nếu có), theo thứ tự cfg['classes']."""
    import torch
    clean = preprocess_phobert(title, content)
    enc   = head_tail_encode(clean, tokenizer, device)

    with torch.no_grad():
        logits = model(**enc).logits[0].float().cpu().numpy()

    # Temperature Scaling -> calibrated softmax
    logits_s = logits / max(float(temperature), 1e-3)
    exp_s = np.exp(logits_s - logits_s.max())
    probs = exp_s / exp_s.sum()

    if thresholds is not None:
        adj = probs / thresholds
        adj /= adj.sum()
        return adj
    return probs


# ── Ensemble (điểm tin cậy kết hợp) ─────────────────────────────────────────
def ensemble_predict(title: str, content: str,
                     pipeline: dict,
                     tokenizer, pho_model, pho_cfg: dict,
                     thresholds, temperature: float, device: str) -> dict:
    """
    Kết hợp 2 mô hình bằng điểm tin cậy:
        Score(c | SVM, BERT) ∝ Score_SVM(c) × Score_BERT(c)
    Khi 2 model cùng cho score cao ở class c → score kết hợp càng lớn.
    """
    if not title.strip() and not content.strip():
        raise ValueError("Không có nội dung để phân loại.")

    classes_svm = pipeline["classes"]
    classes_pho = pho_cfg["classes"]

    svm_probs = _predict_svm(title, content, pipeline)
    pho_probs = _predict_phobert(title, content, tokenizer, pho_model,
                                 pho_cfg, thresholds, temperature, device)

    svm_dict = {cls: float(p) for cls, p in zip(classes_svm, svm_probs)}
    pho_dict = {cls: float(p) for cls, p in zip(classes_pho, pho_probs)}

    # Union classes
    all_classes = sorted(set(classes_svm) | set(classes_pho))

    # Tích điểm tin cậy theo từng class
    raw = {cls: svm_dict.get(cls, 1e-9) * pho_dict.get(cls, 1e-9) for cls in all_classes}
    total    = sum(raw.values())
    combined = {cls: p / total for cls, p in raw.items()}

    top_sorted = sorted(combined.items(), key=lambda x: -x[1])
    pred_class = top_sorted[0][0]

    svm_pred = classes_svm[int(np.argmax(svm_probs))]
    pho_pred = classes_pho[int(np.argmax(pho_probs))]
    agree    = (svm_pred == pho_pred)

    return {
        "pred_class":      pred_class,
        "confidence":      combined[pred_class],
        "top5":            top_sorted[:5],
        "all_probs":       combined,
        "agree":           agree,
        "svm_pred":        svm_pred,
        "svm_conf":        svm_dict.get(svm_pred, 0.0),
        "svm_probs":       svm_dict,
        "pho_pred":        pho_pred,
        "pho_conf":        pho_dict.get(pho_pred, 0.0),
        "pho_probs":       pho_dict,
        "pho_model_name":  pho_cfg.get("model_name", "PhoBERT"),
        "title":           title,
        "content_preview": content[:600] + ("…" if len(content) > 600 else ""),
    }


# ── Top keywords SVM ──────────────────────────────────────────────────────────
def get_top_keywords(pipeline: dict, pred_class: str, n: int = 8) -> list:
    classes = pipeline["classes"]
    if pred_class not in classes or not hasattr(pipeline["model"], "coef_"):
        return []
    class_idx     = classes.index(pred_class)
    coef          = pipeline["model"].coef_[class_idx]
    feature_names = pipeline["vectorizer"].get_feature_names_out()
    top_idx       = np.argsort(coef)[::-1][:n]
    return [(feature_names[i], float(coef[i])) for i in top_idx]


# ── Hiển thị kết quả ensemble ─────────────────────────────────────────────────
def show_result(res: dict, pipeline: dict, source_label: str = ""):
    conf_pct = res["confidence"] * 100
    agree    = res["agree"]

    if agree:
        st.success(
            f"**Hai mô hình đồng thuận:** cả SVM lẫn PhoBERT đều dự đoán "
            f"**{res['pred_class']}** — độ tin cậy kết hợp cao hơn đáng kể.",
        )
    else:
        st.warning(
            f"**Hai mô hình bất đồng:** SVM → **{res['svm_pred']}**  |  "
            f"PhoBERT → **{res['pho_pred']}**  |  "
            f"Kết quả ensemble: **{res['pred_class']}** ({conf_pct:.1f}%)"
        )

    if conf_pct >= 60:
        color, badge = "#27ae60", "Tin cậy cao"
    elif conf_pct >= 40:
        color, badge = "#e67e22", "Tin cậy trung bình"
    else:
        color, badge = "#e74c3c", "Tin cậy thấp"

    agree_note = " · 🤝 Đồng thuận" if agree else " · ⚡ Ensemble"

    col_main, col_models, col_content = st.columns([2, 2, 2])

    # Cột 1: Kết quả tổng hợp
    with col_main:
        st.subheader("Kết quả Ensemble")
        st.markdown(
            f"""<div style='background:{color}18;border-left:5px solid {color};
                           padding:16px 20px;border-radius:8px;margin-bottom:14px'>
                    <div style='font-size:1.5rem;font-weight:800;color:{color};
                                margin-bottom:4px'>{res["pred_class"]}</div>
                    <div style='color:#555;font-size:.9rem'>
                        Tin cậy: <b>{conf_pct:.1f}%</b>&nbsp;·&nbsp;{badge}{agree_note}
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Top 5 chủ đề (combined):**")
        for i, (cls, prob) in enumerate(res["top5"]):
            label = f"🥇 **{cls}**" if i == 0 else f"{i+1}. {cls}"
            st.markdown(f"{label}&nbsp;`{prob*100:.1f}%`", unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown("**Top từ khoá (SVM):**")
        kws = get_top_keywords(pipeline, res["pred_class"])
        if kws:
            df_kws = pd.DataFrame(kws, columns=["Từ / Bigram", "Coef"])
            df_kws.index += 1
            st.dataframe(
                df_kws.style.format({"Coef": "{:.4f}"}),
                use_container_width=True, height=260,
            )

    # Cột 2: So sánh từng model
    with col_models:
        st.subheader("Từng mô hình")

        svm_pct = res["svm_conf"] * 100
        svm_col = "#27ae60" if res["svm_pred"] == res["pred_class"] else "#e67e22"
        st.markdown(
            f"""<div style='background:{svm_col}12;border:1px solid {svm_col}55;
                           padding:12px 16px;border-radius:8px;margin-bottom:10px'>
                    <div style='font-weight:700;color:{svm_col}'>SVM + TF-IDF</div>
                    <div style='font-size:1.1rem;font-weight:600'>{res["svm_pred"]}</div>
                    <div style='color:#666;font-size:.85rem'>Điểm tin cậy suy ra: {svm_pct:.1f}%</div>
                </div>""",
            unsafe_allow_html=True,
        )

        pho_pct = res["pho_conf"] * 100
        pho_col = "#27ae60" if res["pho_pred"] == res["pred_class"] else "#e67e22"
        st.markdown(
            f"""<div style='background:{pho_col}12;border:1px solid {pho_col}55;
                           padding:12px 16px;border-radius:8px;margin-bottom:10px'>
                    <div style='font-weight:700;color:{pho_col}'>{res.get("pho_model_name", "PhoBERT")}</div>
                    <div style='font-size:1.1rem;font-weight:600'>{res["pho_pred"]}</div>
                    <div style='color:#666;font-size:.85rem'>Điểm sau calibration: {pho_pct:.1f}%</div>
                </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Top 5 — so sánh:**")
        top5_cls = [c for c, _ in res["top5"]]
        rows = []
        for cls in top5_cls:
            rows.append({
                "Chủ đề":     cls,
                "SVM %":      f"{res['svm_probs'].get(cls, 0)*100:.1f}",
                "PhoBERT %":  f"{res['pho_probs'].get(cls, 0)*100:.1f}",
                "Combined %": f"{res['all_probs'].get(cls, 0)*100:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Cột 3: Nội dung bài báo
    with col_content:
        st.subheader("Nội dung bài báo")
        if res["title"]:
            st.markdown(f"**Tiêu đề:** {res['title']}")
        st.text_area(
            "_content",
            value=res["content_preview"] or "(không có nội dung)",
            height=460,
            disabled=True,
            label_visibility="collapsed",
        )

    # Lưu lịch sử
    st.session_state.history.insert(0, {
        "Thời gian":  datetime.datetime.now().strftime("%H:%M:%S"),
        "Nguồn":      source_label[:70] or res["title"][:70] or "—",
        "Ensemble":   res["pred_class"],
        "SVM":        res["svm_pred"],
        "PhoBERT":    res["pho_pred"],
        "Đồng thuận": "✅" if res["agree"] else "❌",
        "Tin cậy":    f"{conf_pct:.1f}%",
    })


# ── Tab Lịch sử ───────────────────────────────────────────────────────────────
def show_history():
    if not st.session_state.get("history"):
        st.info("Chưa có lịch sử phân loại trong session này.")
        return

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, hide_index=True, use_container_width=True)

    col_dl, col_clr, _ = st.columns([2, 2, 6])
    with col_dl:
        st.download_button(
            "Tải CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="history_combined.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clr:
        if st.button("Xoá lịch sử", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Phân loại tin tức — Ensemble",
        page_icon="🇻🇳",
        layout="wide",
    )

    st.title("🇻🇳 Phân loại tin tức Vietnamnet — Ensemble SVM + PhoBERT")
    st.markdown(
            "Kết hợp **SVM + TF-IDF** và **PhoBERT** bằng **điểm tin cậy kết hợp**. "
            "Khi hai mô hình đồng thuận, kết quả đáng tin cậy hơn đáng kể."
    )

    # ── Load models ───────────────────────────────────────────────────────────
    errors = []
    try:
        pipeline = load_svm()
    except Exception as e:
        pipeline = None
        errors.append(f"❌ Không load được SVM: `{e}`  →  Chạy `main_SVM.ipynb` trước.")

    try:
        tokenizer, pho_model, pho_cfg, thresholds, temperature, device = load_phobert()
    except Exception as e:
        tokenizer = pho_model = pho_cfg = thresholds = temperature = device = None
        errors.append(f"❌ Không load được PhoBERT: `{e}`  →  Chạy `main_PhoBERT.ipynb` trước.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Thông tin mô hình")
        st.caption(
            f"**SVM:** `inference_pipeline.pkl`\n\n"
            f"Classes: {len(pipeline['classes'])}  \n"
            f"Max features: {pipeline['config'].get('max_features', '?'):,}"
        )
        st.divider()
        st.caption(
            f"**{pho_cfg.get('model_name', 'PhoBERT')}:** `PhoBERT/model/`\n\n"
            f"Classes: {len(pho_cfg['classes'])}  \n"
            f"Device: {device.upper()}  \n"
            f"Threshold calibration: {'BẬT' if thresholds is not None else 'tắt'}"
        )
        st.divider()
        st.markdown(
            "**Công thức ensemble:**\n\n"
            "`Score(c|SVM,BERT) ∝ Score_SVM(c) × Score_BERT(c)`\n\n"
            "Nhân điểm tin cậy của cả hai mô hình, sau đó chuẩn hóa về tổng = 1."
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 Nhập URL", "📝 Nhập text", "📋 Batch URL", "📜 Lịch sử",
    ])

    # ── Tab 1: URL ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("Nhập URL bài báo rồi nhấn **Phân loại**.")
        with st.form("form_url", border=False):
            url = st.text_input("URL", placeholder="https://vietnamnet.vn/...",
                                label_visibility="collapsed")
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
                        st.info("Dùng tab **📝 Nhập text** để dán nội dung thủ công.")
                        st.stop()

                if not title and not content:
                    st.error("❌ Không tìm thấy nội dung — thử tab **📝 Nhập text**.")
                else:
                    with st.spinner("Đang chạy SVM + PhoBERT..."):
                        res = ensemble_predict(
                            title, content, pipeline,
                            tokenizer, pho_model, pho_cfg, thresholds, temperature, device,
                        )
                    st.divider()
                    show_result(res, pipeline, source_label=url.strip())

    # ── Tab 2: Text ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Dùng khi URL bị chặn (paywall, Cloudflare…).")
        with st.form("form_text", border=False):
            t_title   = st.text_input("Tiêu đề", placeholder="Nhập hoặc dán tiêu đề…")
            t_content = st.text_area("Nội dung", placeholder="Dán nội dung bài báo…",
                                     height=220)
            sub2 = st.form_submit_button("Phân loại", type="primary")

        if sub2:
            if not t_title.strip() and not t_content.strip():
                st.warning("Vui lòng nhập ít nhất tiêu đề hoặc nội dung.")
            else:
                with st.spinner("Đang chạy SVM + PhoBERT..."):
                    res2 = ensemble_predict(
                        t_title.strip(), t_content.strip(), pipeline,
                        tokenizer, pho_model, pho_cfg, thresholds, temperature, device,
                    )
                st.divider()
                show_result(res2, pipeline,
                            source_label=t_title.strip()[:70] or "text input")

    # ── Tab 3: Batch ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("Dán nhiều URL (mỗi dòng 1 URL) — phân loại hàng loạt.")
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
                results = []
                prog = st.progress(0.0, text="Đang xử lý…")

                for idx, u in enumerate(urls_list):
                    prog.progress(
                        (idx + 1) / len(urls_list),
                        text=f"[{idx+1}/{len(urls_list)}] {u[:60]}…",
                    )
                    try:
                        _t, _c = scrape_article(u)
                        _r = ensemble_predict(
                            _t, _c, pipeline,
                            tokenizer, pho_model, pho_cfg, thresholds, temperature, device,
                        )
                        results.append({
                            "URL":        u,
                            "Tiêu đề":    _t[:80],
                            "Ensemble":   _r["pred_class"],
                            "SVM":        _r["svm_pred"],
                            "PhoBERT":    _r["pho_pred"],
                            "Đồng thuận": "✅" if _r["agree"] else "❌",
                            "Tin cậy":    f"{_r['confidence']*100:.1f}%",
                            "Trạng thái": "✅ OK",
                        })
                        st.session_state.history.insert(0, {
                            "Thời gian":  datetime.datetime.now().strftime("%H:%M:%S"),
                            "Nguồn":      u[:70],
                            "Ensemble":   _r["pred_class"],
                            "SVM":        _r["svm_pred"],
                            "PhoBERT":    _r["pho_pred"],
                            "Đồng thuận": "✅" if _r["agree"] else "❌",
                            "Tin cậy":    f"{_r['confidence']*100:.1f}%",
                        })
                    except Exception as e:
                        results.append({
                            "URL":        u, "Tiêu đề": "—",
                            "Ensemble":   "—", "SVM": "—", "PhoBERT": "—",
                            "Đồng thuận": "—", "Tin cậy": "—",
                            "Trạng thái": f"❌ {e}",
                        })

                prog.empty()
                df_batch = pd.DataFrame(results)
                ok  = (df_batch["Trạng thái"] == "✅ OK").sum()
                agr = (df_batch["Đồng thuận"] == "✅").sum()
                st.success(
                    f"Hoàn thành: {ok}/{len(urls_list)} URL thành công  |  "
                    f"Đồng thuận: {agr}/{ok}"
                )
                st.dataframe(df_batch, hide_index=True, use_container_width=True)
                st.download_button(
                    "Tải kết quả (CSV)",
                    data=df_batch.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="batch_ensemble.csv",
                    mime="text/csv",
                )

    # ── Tab 4: Lịch sử ────────────────────────────────────────────────────────
    with tab4:
        show_history()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("19 chủ đề"):
        cols = st.columns(4)
        for i, cls in enumerate(pipeline["classes"]):
            cols[i % 4].markdown(f"- {cls}")


if __name__ == "__main__":
    main()
