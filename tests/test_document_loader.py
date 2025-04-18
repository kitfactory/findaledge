import sys
from pathlib import Path
import os
import shutil # テストディレクトリ作成用
import pytest

from langchain.schema import Document as LangchainDocument
from findaledge.document_loader import DocumentLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# テスト用データディレクトリのパス
TEST_DATA_DIR = Path(__file__).parent / "data" / "document_loader"
SUB_DIR = TEST_DATA_DIR / "subdir"
# Add directory with special characters
SPECIAL_CHARS_DIR = TEST_DATA_DIR / "ディレクトリ名 スペース 日本語"

# --- Test Setup ---
@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Create dummy files for testing."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)
    SPECIAL_CHARS_DIR.mkdir(exist_ok=True) # Create dir with special chars

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
        # Add empty files for supported types
        (TEST_DATA_DIR / "empty.txt"): "",
        (TEST_DATA_DIR / "empty.md"): "",
        (TEST_DATA_DIR / "empty.html"): "",
        (TEST_DATA_DIR / "empty.py"): "",
        (TEST_DATA_DIR / "empty.js"): "",
        (TEST_DATA_DIR / "empty.json"): "",
        # Add file with non-ASCII characters
        (TEST_DATA_DIR / "encoding_test.txt"): "これは日本語のテストです。",
        # Add files with special characters in path
        (SPECIAL_CHARS_DIR / "ファイル名 スペース.txt"): "Content in file with special name.",
        (SPECIAL_CHARS_DIR / "another file.md"): "# Another Markdown",
        # Add file with invalid encoding (for testing error handling)
        # Note: This might be None or specific bytes depending on how we create it
        (TEST_DATA_DIR / "invalid_encoding.txt"): b'\x80abc', # Invalid UTF-8 start byte
    }

    for file_path, content in files_to_create.items():
        # Ensure parent directory exists before creating file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
             if isinstance(content, bytes):
                 # Write bytes directly for encoding test
                 with open(file_path, "wb") as f:
                     f.write(content)
             elif content is not None:
                  # Explicitly use utf-8 encoding for writing text
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

def test_load_directory_recursive(loader: DocumentLoader, capsys):
    """Test loading from a directory recursively, checking skips."""
    # Expected loadable files based on the setup and updated loader logic:
    # - text/markup/code files with valid content/encoding
    # - empty files of supported types
    # - files in special character directory
    # Excluded: unsupported.xyz, invalid_encoding.txt (should now return None)
    # PDF/DOCX might load depending on markitdown/dependencies, but we focus on known good/bad cases.
    expected_sources = {
        str(TEST_DATA_DIR / "test.txt"),
        str(TEST_DATA_DIR / "test.md"),
        str(TEST_DATA_DIR / "test.html"),
        str(SUB_DIR / "sub_test.txt"),
        str(TEST_DATA_DIR / "script.py"),
        str(TEST_DATA_DIR / "script.js"),
        str(TEST_DATA_DIR / "data.json"),
        str(TEST_DATA_DIR / "Dockerfile"),
        str(TEST_DATA_DIR / "empty.txt"),
        str(TEST_DATA_DIR / "empty.md"),
        str(TEST_DATA_DIR / "empty.html"),
        str(TEST_DATA_DIR / "empty.py"),
        str(TEST_DATA_DIR / "empty.js"),
        str(TEST_DATA_DIR / "empty.json"),
        str(TEST_DATA_DIR / "encoding_test.txt"),
        str(SPECIAL_CHARS_DIR / "ファイル名 スペース.txt"),
        str(SPECIAL_CHARS_DIR / "another file.md"),
    }
    # Add pdf/docx if they are expected to load successfully (even if empty)
    # If empty.pdf/empty.docx exist and markitdown loads them:
    if (TEST_DATA_DIR / "empty.pdf").exists() and (TEST_DATA_DIR / "empty.pdf").stat().st_size > 0: # Adjust check if empty files load
        expected_sources.add(str(TEST_DATA_DIR / "empty.pdf"))
    if (TEST_DATA_DIR / "empty.docx").exists() and (TEST_DATA_DIR / "empty.docx").stat().st_size > 0:
        expected_sources.add(str(TEST_DATA_DIR / "empty.docx"))

    expected_loadable_count = len(expected_sources)

    docs = loader.load_from_directory(TEST_DATA_DIR) # Recursive by default
    sources = {doc.metadata["source"] for doc in docs}

    # Assert all expected loadable files are present
    assert sources == expected_sources # Check for exact match now

    # Assert known problematic files were skipped and messages printed
    captured = capsys.readouterr()
    unsupported_file_path = str(TEST_DATA_DIR / "unsupported.xyz")
    invalid_encoding_file_path = str(TEST_DATA_DIR / "invalid_encoding.txt")

    # Check for skip message for unsupported file
    assert f"Skipping unsupported file type: {unsupported_file_path}" in captured.out

    # Check for warning message for invalid encoding file (should fail UTF-8 read)
    assert f"[WARN] UTF-8 decoding failed for file {invalid_encoding_file_path}. Skipping." in captured.out

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
    # Expected: test.txt, sub_test.txt, empty.txt, encoding_test.txt, ファイル名 スペース.txt
    # invalid_encoding.txt has .txt but should be skipped due to invalid content/markitdown error
    expected_files = [
        str(TEST_DATA_DIR / "test.txt"),
        str(SUB_DIR / "sub_test.txt"),
        str(TEST_DATA_DIR / "empty.txt"),
        str(TEST_DATA_DIR / "encoding_test.txt"),
        str(SPECIAL_CHARS_DIR / "ファイル名 スペース.txt"),
    ]
    docs = loader.load_from_directory(TEST_DATA_DIR, glob_pattern="**/*.txt", recursive=True)
    assert len(docs) == len(expected_files)
    sources = {doc.metadata["source"] for doc in docs}
    assert sources == set(expected_files)
    assert str(TEST_DATA_DIR / "test.md") not in sources
    assert str(TEST_DATA_DIR / "invalid_encoding.txt") not in sources # Verify skipped

def test_load_directory_non_recursive_glob(loader: DocumentLoader):
    """Test loading non-recursively with glob."""
    docs = loader.load_from_directory(TEST_DATA_DIR, glob_pattern="*.md", recursive=False)
    # Expected: test.md, empty.md
    assert len(docs) == 2 # Corrected expectation
    sources = {doc.metadata["source"] for doc in docs}
    assert str(TEST_DATA_DIR / "test.md") in sources
    assert str(TEST_DATA_DIR / "empty.md") in sources

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

# --- Tests for Empty Files ---

def test_load_empty_txt_file(loader: DocumentLoader):
    """Test loading an empty .txt file."""
    file_path = TEST_DATA_DIR / "empty.txt"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert docs.page_content == ""
    assert docs.metadata["source"] == str(file_path)

def test_load_empty_md_file(loader: DocumentLoader):
    """Test loading an empty .md file."""
    file_path = TEST_DATA_DIR / "empty.md"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert docs.page_content == ""
    assert docs.metadata["source"] == str(file_path)

def test_load_empty_html_file(loader: DocumentLoader):
    """Test loading an empty .html file."""
    file_path = TEST_DATA_DIR / "empty.html"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    # markitdown might produce minimal structure or empty string for empty html
    assert docs.page_content is not None # Check content exists
    assert docs.metadata["source"] == str(file_path)

def test_load_empty_py_file(loader: DocumentLoader):
    """Test loading an empty .py file."""
    file_path = TEST_DATA_DIR / "empty.py"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert docs.page_content == ""
    assert docs.metadata["source"] == str(file_path)

def test_load_empty_js_file(loader: DocumentLoader):
    """Test loading an empty .js file."""
    file_path = TEST_DATA_DIR / "empty.js"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert docs.page_content == ""
    assert docs.metadata["source"] == str(file_path)

def test_load_empty_json_file(loader: DocumentLoader):
    """Test loading an empty .json file."""
    file_path = TEST_DATA_DIR / "empty.json"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert docs.page_content == "" # Loaded as text
    assert docs.metadata["source"] == str(file_path)

# --- Test Special Characters in Path ---

def test_load_file_with_special_chars_in_path(loader: DocumentLoader):
    """Test loading a file with spaces and non-ASCII chars in its path."""
    file_path_space = SPECIAL_CHARS_DIR / "ファイル名 スペース.txt"
    file_path_md = SPECIAL_CHARS_DIR / "another file.md"

    # Test file 1
    docs1 = loader.load_file(file_path_space)
    assert docs1 is not None
    assert isinstance(docs1, LangchainDocument)
    assert "Content in file with special name." in docs1.page_content
    assert docs1.metadata["source"] == str(file_path_space)

    # Test file 2
    docs2 = loader.load_file(file_path_md)
    assert docs2 is not None
    assert isinstance(docs2, LangchainDocument)
    assert "Another Markdown" in docs2.page_content
    assert docs2.metadata["source"] == str(file_path_md)


def test_load_directory_with_special_chars_in_path(loader: DocumentLoader):
    """Test loading from a directory containing files with special path characters."""
    docs = loader.load_from_directory(SPECIAL_CHARS_DIR, recursive=True)
    assert len(docs) == 2 # Expecting the two files created in SPECIAL_CHARS_DIR
    sources = {doc.metadata["source"] for doc in docs}
    assert str(SPECIAL_CHARS_DIR / "ファイル名 スペース.txt") in sources
    assert str(SPECIAL_CHARS_DIR / "another file.md") in sources

# --- Test Encoding ---
def test_load_encoding_file(loader: DocumentLoader):
    """Test loading a file with UTF-8 encoded characters."""
    file_path = TEST_DATA_DIR / "encoding_test.txt"
    docs = loader.load_file(file_path)
    assert docs is not None
    assert isinstance(docs, LangchainDocument)
    assert "これは日本語のテストです。" in docs.page_content
    assert docs.metadata["source"] == str(file_path)

def test_load_invalid_encoding_file(loader: DocumentLoader, capsys):
    """Test loading a file with invalid UTF-8 encoding."""
    file_path = TEST_DATA_DIR / "invalid_encoding.txt"
    docs = loader.load_file(file_path)
    assert docs is None # Expecting the loader to skip or fail gracefully
    captured = capsys.readouterr()
    # Check for the specific warning message from the UTF-8 pre-check
    assert f"[WARN] UTF-8 decoding failed for file {file_path}. Skipping." in captured.out 