from pydantic import BaseModel


class VectorRecord(BaseModel):
    embedding: list[float]
    is_correct: bool = True


class UpdateUserBaselineRequest(BaseModel):
    device_id: str
    username: str
    vectors: list[VectorRecord] = []


class UpdateUserBaselineResponse(BaseModel):
    status: str
    action: str
    username: str
    max_similarity: float


class ThresholdResponse(BaseModel):
    device_id: str
    optimal_threshold: float
    sample_count: int


class DeviceVectorEntry(BaseModel):
    username: str
    vectors: list[list[float]]
    vector_count: int


class GetVectorsByDeviceIdResponse(BaseModel):
    device_id: str
    users: list[DeviceVectorEntry]
    total_users: int
