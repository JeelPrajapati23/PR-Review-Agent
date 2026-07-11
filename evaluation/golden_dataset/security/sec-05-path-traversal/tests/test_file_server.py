from src.file_server import read_user_file


def test_reads_a_file_inside_the_base_directory(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"hello world")

    content = read_user_file(str(tmp_path), "notes.txt")

    assert content == b"hello world"
