from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import grpc

from app.core.model_loader import ArtifactBackedClassifier
from nlp.artifacts import PROJECT_ROOT
from app.generated import classifier_pb2


class ClassifierGrpcService:
    service_name = "vnn.classifier.ClassifierService"

    def __init__(self, classifier: ArtifactBackedClassifier):
        self._classifier = classifier

    def classify_article(
        self,
        request: classifier_pb2.ClassifyRequest,
        context: grpc.ServicerContext,
    ) -> classifier_pb2.ClassifyResponse:
        if not 1 <= request.top_k <= 5:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "top_k must be between 1 and 5")
        from nlp.preprocessing import prepare_text
        if not prepare_text(request.title, request.content):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Article has no usable text after preprocessing")
        response = self._classifier.predict(
            {
                "request_id": request.request_id,
                "title": request.title,
                "content": request.content,
                "source_url": request.source_url,
                "top_k": request.top_k,
            }
        )
        return classifier_pb2.ClassifyResponse(
            request_id=response["request_id"],
            model_version=response["model_version"],
            label=response["label"],
            confidence=response["confidence"],
            margin=response["margin"],
            candidates=[
                classifier_pb2.Candidate(label=item["label"], score=item["score"], decision_score=item["decision_score"])
                for item in response["candidates"]
            ],
            rationale_keywords=response["rationale_keywords"],
            auto_decision=response["auto_decision"],
            latency_ms=response["latency_ms"],
        )

    def validate_artifact(self, request, context):
        try:
            from pathlib import Path
            relative = Path(request.artifact_path)
            directory = (PROJECT_ROOT / relative).resolve()
            if relative.is_absolute() or not directory.is_relative_to(PROJECT_ROOT):
                raise ValueError("Artifact path must stay inside the project")
            candidate = ArtifactBackedClassifier(directory)
            candidate.predict({"title": "Kiểm tra mô hình", "content": "Phân loại tin tức tiếng Việt.", "top_k": 1})
            return classifier_pb2.ValidateArtifactResponse(ok=True)
        except Exception as exc:
            return classifier_pb2.ValidateArtifactResponse(ok=False, error=str(exc))

    def build_server(self, host: str, port: int, workers: int) -> grpc.Server:
        server = grpc.server(ThreadPoolExecutor(max_workers=workers))
        handler = grpc.unary_unary_rpc_method_handler(
            self.classify_article,
            request_deserializer=classifier_pb2.ClassifyRequest.FromString,
            response_serializer=lambda payload: payload.SerializeToString(),
        )
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    self.service_name,
                    {"ClassifyArticle": handler, "ValidateArtifact": grpc.unary_unary_rpc_method_handler(self.validate_artifact, request_deserializer=classifier_pb2.ValidateArtifactRequest.FromString, response_serializer=lambda payload: payload.SerializeToString())},
                ),
            )
        )
        server.add_insecure_port(f"{host}:{port}")
        return server
