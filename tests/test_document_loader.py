import pytest
from pathlib import Path
import os
import shutil # テストディレクトリ作成用

from langchain.schema import Document as LangchainDocument
from finderledge.document_loader import DocumentLoader

# テスト用データディレクトリのパス
TEST_DATA_DIR = Path(__file__).parent / "data" / "document_loader"
SUB_DIR = TEST_DATA_DIR / "subdir"

# --- Test Setup ---
@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Create dummy files for testing."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)

    # Create dummy files if they don't exist
    files_to_create = {
        TEST_DATA_DIR / "test.txt": "This is a test text file.",
        TEST_DATA_DIR / "test.md": "# Test Markdown\n\nThis is a paragraph.",
        TEST_DATA_DIR / "test.html": "<html><body><h1>Test HTML</h1><p>A paragraph.</p></body></html>",
        TEST_DATA_DIR / "unsupported.xyz": "This file type is not supported.",
        SUB_DIR / "sub_test.txt": "This is text in a subdirectory.",
        # Add code files
        TEST_DATA_DIR / "script.py": "import os\nprint(\"Hello Python!\")",
        TEST_DATA_DIR / "script.js": "console.log('Hello JavaScript!');",
        TEST_DATA_DIR / "data.json": "{\"key\": \"value\", \"number\": 123}",
        TEST_DATA_DIR / "Dockerfile": "FROM python:3.11-slim\nCOPY . /app",

        # Add minimal binary files if possible/necessary for basic loading checks
        (TEST_DATA_DIR / "empty.pdf"): None, # Use touch
        (TEST_DATA_DIR / "empty.docx"): None, # Use touch
    }

    for file_path, content in files_to_create.items():
        if not file_path.exists():
             if content is not None:
                  with open(file_path, "w", encoding="utf-8") as f:
                      f.write(content)
             else:
                 file_path.touch(exist_ok=True)


    yield # Run tests

    # --- Teardown (optional: remove created files/dirs) ---
    # Uncomment below to clean up test files after tests run
    # shutil.rmtree(TEST_DATA_DIR)

@pytest.fixture
def loader() -> DocumentLoader:
    """Fixture for DocumentLoader instance."""
    return DocumentLoader()

# --- Test Cases ---

def test_load_txt_file(loader: DocumentLoader):
    """Test loading a .txt file."""
    file_path = TEST_DATA_DIR / "test.txt"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "This is a test text file." in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_md_file(loader: DocumentLoader):
    """Test loading a .md file."""
    file_path = TEST_DATA_DIR / "test.md"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    # markitdown might add formatting, check for core content
    assert "Test Markdown" in docs.page_content
    assert "This is a paragraph." in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_html_file(loader: DocumentLoader):
    """Test loading an .html file."""
    file_path = TEST_DATA_DIR / "test.html"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    # markitdown converts HTML to Markdown
    assert "Test HTML" in docs.page_content # Check heading
    assert "A paragraph." in docs.page_content # Check paragraph
    assert docs.metadata["source"] == str(file_path)

# --- Tests for potentially complex formats (basic check) ---
# These tests assume markitdown and dependencies are installed
# They primarily check if loading succeeds without error

# @pytest.mark.skipif(not (TEST_DATA_DIR / "test.pdf").exists(), reason="Test PDF file not found")
def test_load_pdf_file_basic(loader: DocumentLoader):
    """Basic test loading a .pdf file (checks for success)."""
    # Replace 'empty.pdf' with an actual simple PDF for content check if available
    file_path = TEST_DATA_DIR / "empty.pdf" # Use empty for now
    if not file_path.exists() or file_path.stat().st_size == 0:
         pytest.skip(f"Skipping PDF test, {file_path} is empty or missing.")

    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert isinstance(docs.page_content, str) # Content might be empty for empty file
    assert docs.metadata["source"] == str(file_path)


# @pytest.mark.skipif(not (TEST_DATA_DIR / "test.docx").exists(), reason="Test DOCX file not found")
def test_load_docx_file_basic(loader: DocumentLoader):
    """Basic test loading a .docx file (checks for success)."""
     # Replace 'empty.docx' with an actual simple DOCX for content check if available
    file_path = TEST_DATA_DIR / "empty.docx" # Use empty for now
    if not file_path.exists() or file_path.stat().st_size == 0:
         pytest.skip(f"Skipping DOCX test, {file_path} is empty or missing.")

    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert isinstance(docs.page_content, str)
    assert docs.metadata["source"] == str(file_path)

# --- Test unsupported file ---

def test_load_unsupported_file(loader: DocumentLoader, capsys):
    """Test loading an unsupported file type."""
    file_path = TEST_DATA_DIR / "unsupported.xyz"
    docs = loader.load_file(file_path)
    assert docs is None
    captured = capsys.readouterr()
    assert f"Skipping unsupported file type: {file_path}" in captured.out


def test_load_nonexistent_file(loader: DocumentLoader):
    """Test loading a non-existent file."""
    file_path = TEST_DATA_DIR / "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        loader.load_file(file_path)

# --- Test load_from_directory ---

def test_load_directory_recursive(loader: DocumentLoader):
    """Test loading from a directory recursively."""
    # Update expected count: 4 (original) + 4 (code files) = 8
    # Adjust if pdf/docx tests are skipped
    expected_min_count = 8
    docs = loader.load_from_directory(TEST_DATA_DIR)
    assert len(docs) >= expected_min_count
    sources = {doc.metadata["source"] for doc in docs}
    assert str(TEST_DATA_DIR / "test.txt") in sources
    assert str(TEST_DATA_DIR / "test.md") in sources
    assert str(TEST_DATA_DIR / "test.html") in sources
    assert str(SUB_DIR / "sub_test.txt") in sources
    assert str(TEST_DATA_DIR / "script.py") in sources
    assert str(TEST_DATA_DIR / "script.js") in sources
    assert str(TEST_DATA_DIR / "data.json") in sources
    assert str(TEST_DATA_DIR / "Dockerfile") in sources
    assert str(TEST_DATA_DIR / "unsupported.xyz") not in sources # Should be skipped

def test_load_directory_non_recursive(loader: DocumentLoader):
    """Test loading from a directory non-recursively."""
    # Update expected count: 3 (original) + 4 (code files) = 7
    # Adjust if pdf/docx tests are skipped
    expected_min_count = 7
    docs = loader.load_from_directory(TEST_DATA_DIR, recursive=False, glob_pattern="*") # Use simpler glob
    assert len(docs) >= expected_min_count
    sources = {doc.metadata["source"] for doc in docs}
    assert str(SUB_DIR / "sub_test.txt") not in sources # Subdirectory file should not be loaded
    assert str(TEST_DATA_DIR / "script.py") in sources
    assert str(TEST_DATA_DIR / "Dockerfile") in sources

def test_load_directory_glob_pattern(loader: DocumentLoader):
    """Test loading using a glob pattern."""
    docs = loader.load_from_directory(TEST_DATA_DIR, glob_pattern="**/*.txt", recursive=True)
    assert len(docs) == 2 # test.txt and sub_test.txt
    sources = {doc.metadata["source"] for doc in docs}
    assert str(TEST_DATA_DIR / "test.txt") in sources
    assert str(SUB_DIR / "sub_test.txt") in sources
    assert str(TEST_DATA_DIR / "test.md") not in sources

def test_load_directory_non_recursive_glob(loader: DocumentLoader):
    """Test loading non-recursively with glob."""
    docs = loader.load_from_directory(TEST_DATA_DIR, glob_pattern="*.md", recursive=False)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(TEST_DATA_DIR / "test.md")

def test_load_directory_nonexistent(loader: DocumentLoader):
    """Test loading from a non-existent directory."""
    dir_path = TEST_DATA_DIR / "nonexistent_dir"
    with pytest.raises(FileNotFoundError):
        loader.load_from_directory(dir_path)

def test_load_directory_recursive_false_with_double_star_glob(loader: DocumentLoader):
    """Test ValueError when recursive=False and glob has '**'."""
    with pytest.raises(ValueError):
        loader.load_from_directory(TEST_DATA_DIR, glob_pattern="**/*.txt", recursive=False)

def test_load_python_file(loader: DocumentLoader):
    """Test loading a .py file as text."""
    file_path = TEST_DATA_DIR / "script.py"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "import os" in docs.page_content
    assert "print(\"Hello Python!\")" in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_javascript_file(loader: DocumentLoader):
    """Test loading a .js file as text."""
    file_path = TEST_DATA_DIR / "script.js"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "console.log('Hello JavaScript!');" in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_json_file(loader: DocumentLoader):
    """Test loading a .json file as text."""
    file_path = TEST_DATA_DIR / "data.json"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "{\"key\": \"value\", \"number\": 123}" in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_dockerfile(loader: DocumentLoader):
    """Test loading a Dockerfile (no extension) as text."""
    file_path = TEST_DATA_DIR / "Dockerfile"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "FROM python:3.11-slim" in docs.page_content
    assert "COPY . /app" in docs.page_content
    assert docs.metadata["source"] == str(file_path) 