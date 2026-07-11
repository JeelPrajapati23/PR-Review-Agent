from unittest.mock import MagicMock, patch

from src.archive_tool import compress_directory


def test_compress_directory_invokes_zip_command():
    with patch("src.archive_tool.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = compress_directory("reports")

        assert exit_code == 0
        called_command = mock_run.call_args[0][0]
        assert "reports" in called_command


def test_compress_directory_returns_nonzero_on_failure():
    with patch("src.archive_tool.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)

        exit_code = compress_directory("missing_dir")

        assert exit_code == 1
