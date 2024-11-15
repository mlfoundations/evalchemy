# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

from datetime import datetime
from unittest.mock import MagicMock, patch

from github import Auth, Github
from github.ContentFile import ContentFile

from camel.toolkits.github_toolkit import (
    GithubIssue,
    GithubPullRequest,
    GithubPullRequestDiff,
    GithubToolkit,
)


@patch.object(Github, "__init__", lambda self, *args, **kwargs: None)
@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_init(mock_get_repo):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    # Assert that the get_repo method was called with the correct argument
    github_toolkit.github.get_repo.assert_called_once_with("repo_name")


@patch.object(Github, "__init__", lambda self, *args, **kwargs: None)
@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_get_tools(mock_get_repo):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    tools = github_toolkit.get_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0


@patch.object(Github, "__init__", lambda self, *args, **kwargs: None)
@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_retrieve_issue_list(monkeypatch):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    # Create a mock issue object
    mock_issue = MagicMock()
    mock_issue.number = 1
    mock_issue.title = "Test Issue"
    mock_issue.body = "This is a test issue"
    mock_issue.pull_request = False

    mock_label = MagicMock()
    mock_label.name = "path/to/file"
    mock_issue.labels = [mock_label]

    # Mock the get_issues method of the mock_repo instance to return a list
    # containing the mock issue object
    github_toolkit.repo.get_issues.return_value = [mock_issue]
    github_toolkit.retrieve_file_content = MagicMock(return_value="This is the content of the file")

    # Call the retrieve_issue_list method
    issue_list = github_toolkit.retrieve_issue_list()

    # Assert the returned issue list
    expected_issue = GithubIssue(
        title="Test Issue",
        body="This is a test issue",
        number=1,
        file_path="path/to/file",
        file_content="This is the content of the file",
    )
    assert issue_list == [expected_issue], f"Expected {expected_issue}, but got {issue_list}"


@patch.object(Github, "__init__", lambda self, *args, **kwargs: None)
@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_retrieve_issue(monkeypatch):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    # Create a mock issue object
    mock_issue = MagicMock()
    mock_issue.number = 1
    mock_issue.title = "Test Issue"
    mock_issue.body = "This is a test issue"
    mock_issue.pull_request = False

    mock_label = MagicMock()
    mock_label.name = "path/to/file"
    mock_issue.labels = [mock_label]

    # Mock the get_issues method of the mock repo instance to return a list
    # containing the mock issue object
    github_toolkit.repo.get_issues.return_value = [mock_issue]
    github_toolkit.retrieve_file_content = MagicMock(return_value="This is the content of the file")

    # Call the retrieve_issue_list method
    issue = github_toolkit.retrieve_issue(1)

    # Assert the returned issue list
    expected_issue = GithubIssue(
        title="Test Issue",
        body="This is a test issue",
        number=1,
        file_path="path/to/file",
        file_content="This is the content of the file",
    )
    assert issue == str(expected_issue), f"Expected {expected_issue}, but got {issue}"


@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_create_pull_request(monkeypatch):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    # Mock the create_pull method of the github_toolkit instance to return a
    # value
    mock_pr_response = MagicMock()
    mock_pr_response.title = """[GitHub Agent] Solved issue: Time complexity 
    for product_of_array_except_self.py"""
    mock_pr_response.body = "Fixes #1"
    github_toolkit.repo.create_pull.return_value = mock_pr_response

    # Create a MagicMock that mimics ContentFile
    mock_content_file = MagicMock(spec=ContentFile)
    mock_content_file.path = "path/to/file"
    mock_content_file.sha = "dummy_sha"

    # Ensure get_contents returns the mocked ContentFile
    github_toolkit.repo.get_contents.return_value = mock_content_file

    # Create a pull request
    pr = github_toolkit.create_pull_request(
        file_path="path/to/file",
        branch_name="branch_name",
        new_content="This is the content of the file",
        pr_title="""[GitHub Agent] Solved issue: Time complexity for 
        product_of_array_except_self.py""",
        body="Fixes #1",
    )

    expected_response = """Title: [GitHub Agent] Solved issue: Time complexity 
    for product_of_array_except_self.py\nBody: Fixes #1\n"""

    assert pr == expected_response, f"Expected {expected_response}, but got {pr}"


@patch.object(Github, "get_repo", return_value=MagicMock())
@patch.object(Auth.Token, "__init__", lambda self, *args, **kwargs: None)
def test_retrieve_pull_requests(monkeypatch):
    # Call the constructor of the GithubToolkit class
    github_toolkit = GithubToolkit("repo_name", "token")

    # Create a mock file
    mock_file = MagicMock()
    mock_file.filename = "path/to/file"
    mock_file.diff = "This is the diff of the file"

    # Create a mock pull request
    mock_pull_request = MagicMock()
    mock_pull_request.title = "Test PR"
    mock_pull_request.body = "This is a test issue"
    mock_pull_request.merged_at = datetime.utcnow()

    # Create a mock file
    mock_file = MagicMock()
    mock_file.filename = "path/to/file"
    mock_file.patch = "This is the diff of the file"

    # Mock the get_files method of the mock_pull_request instance to return a
    # list containing the mock file object
    mock_pull_request.get_files.return_value = [mock_file]

    # Mock the get_issues method of the mock repo instance to return a list
    # containing the mock issue object
    github_toolkit.repo.get_pulls.return_value = [mock_pull_request]

    pull_requests = github_toolkit.retrieve_pull_requests(days=7, state="closed", max_prs=3)
    # Assert the returned issue list
    expected_pull_request = GithubPullRequest(
        title="Test PR",
        body="This is a test issue",
        diffs=[GithubPullRequestDiff(filename="path/to/file", patch="This is the diff of the file")],
    )
    assert pull_requests == [str(expected_pull_request)], f"Expected {expected_pull_request}, but got {pull_requests}"


def test_github_issue():
    # Create a GithubIssue object
    issue = GithubIssue(
        title="Test Issue",
        body="This is a test issue",
        number=1,
        file_path="path/to/file",
        file_content="This is the content of the file",
    )

    # Assert the attributes of the GithubIssue object
    assert issue.title == "Test Issue"
    assert issue.body == "This is a test issue"
    assert issue.number == 1
    assert issue.file_path == "path/to/file"
    assert issue.file_content == "This is the content of the file"

    # Test the summary method
    summary = str(issue)
    expected_summary = (
        f"Title: {issue.title}\n"
        f"Body: {issue.body}\n"
        f"Number: {issue.number}\n"
        f"File Path: {issue.file_path}\n"
        f"File Content: {issue.file_content}"
    )
    assert summary == expected_summary, f"Expected {expected_summary}, but got {summary}"
