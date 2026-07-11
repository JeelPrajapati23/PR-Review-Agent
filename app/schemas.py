from pydantic import BaseModel


class Repository(BaseModel):
    id: int
    name: str
    full_name: str
    clone_url: str


class Head(BaseModel):
    ref: str
    sha: str


class PullRequest(BaseModel):
    number: int
    title: str
    draft: bool
    head: Head
    modified_files: list[str] = []
    added_files: list[str] = []


class PullRequestEvent(BaseModel):
    action: str
    pull_request: PullRequest
    repository: Repository
