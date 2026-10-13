import grpc
from app.generated import classifier_pb2
with grpc.insecure_channel('localhost:50051') as channel:
    method=channel.unary_unary('/vnn.classifier.ClassifierService/ClassifyArticle',request_serializer=lambda x:x.SerializeToString(),response_deserializer=classifier_pb2.ClassifyResponse.FromString)
    result=method(classifier_pb2.ClassifyRequest(title='Kiểm tra mô hình',content='Phân loại tin tức tiếng Việt bằng mô hình ngôn ngữ.',top_k=1),timeout=25)
    assert result.model_version and result.label
