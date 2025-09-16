import os


def verifyPdf(pdf: str) -> bool:
    """
    Verify whether the provided PDF file exists in the documents directory.
    The function expects `pdf` to be a filename without the `.pdf` extension (keeps parity
    with the existing codebase usage).
    """
    path = os.path.join("./documents", f"{pdf}.pdf")
    return os.path.isfile(path)


def trim_file_name(file_name: str) -> str:
    """
    Trim the file name to remove the `.pdf` extension and validate it.
    """
    if file_name is None:
        raise ValueError("file_name cannot be None")

    trimmed_file_name = file_name.split('.pdf')[0]
    if not trimmed_file_name:
        raise ValueError("File name cannot be empty after trimming.")
    return trimmed_file_name
