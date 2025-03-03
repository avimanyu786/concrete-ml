"""Check that headers linked do indeed exist in target markdown files"""
import os
from pathlib import Path

import mistletoe  # Markdown to AST


def ast_iterator(root):
    """Iterate on all children of a node"""
    nodes = [root]
    while nodes:
        current_node = nodes.pop(0)
        yield current_node
        if hasattr(current_node, "children"):
            nodes += current_node.children


def is_web_link(target: str) -> bool:
    """Check if the link points to http or https"""
    if target.startswith("http://"):
        return True
    if target.startswith("https://"):
        return True
    return False


def is_mailto_link(target: str) -> bool:
    """Check if the link points to a mailto"""
    if "mailto:" in target:
        return True
    return False


def contains_header(ast, header) -> bool:
    """Check if the ast-represented document contains the header"""
    for node in ast_iterator(ast):
        if isinstance(node, mistletoe.block_token.Heading):

            # Heading is list of tokens
            file_header = " ".join(
                [
                    str(elt.content)
                    for elt in ast_iterator(node)
                    if isinstance(elt, mistletoe.span_token.RawText)
                ]
            )
            # Needed to escape some characters
            # We might want to check with the markdown spec
            file_header = (
                "-".join(file_header.split())
                .replace("<kbd>", "")
                .replace("</kbd>", "")
                .replace(".", "")
                .replace("!", "")
                .replace("?", "")
                .lower()
            )

            if header == file_header:
                return True
    return False


# pylint: disable-next=too-many-branches
def main():
    """Main function that checks for all files that the header exists in the linked file"""
    # Get files
    current_path = Path(os.getcwd())
    markdown_files = [
        path
        for path in current_path.rglob("*")
        if str(path).endswith(".md")
        if ".venv" not in set(map(str, path.parts))
    ]

    # Collect ASTs
    asts = {}
    for file_path in markdown_files:
        with open(file_path, mode="r", encoding="utf-8") as file:
            asts[file_path.resolve()] = mistletoe.Document(file)

    # Check links
    errors = []
    # For each document we check all links
    for document_path, document in asts.items():
        for node in ast_iterator(document):
            if isinstance(node, (mistletoe.span_token.Link)):
                # We don't verify external links
                if is_web_link(node.target):
                    continue
                if is_mailto_link(node.target):
                    continue

                # Split file and header
                splitted = node.target.split("#")
                if len(splitted) == 2:
                    if splitted[0]:  # Link to another folder
                        file_path = Path(splitted[0])
                    else:  # Link to self
                        file_path = Path(document_path)
                    header = splitted[1]
                elif len(splitted) == 1:
                    file_path, header = Path(splitted[0]), None
                else:
                    raise ValueError(f"Could not parse {node.target}")

                # Get absolute path
                abs_file_path = (document_path.parent / file_path).resolve()

                # Check file exists
                if not abs_file_path.exists():
                    errors.append(f"{abs_file_path} does not exist")
                    continue

                # Check header is contained
                if header:
                    if abs_file_path not in asts:
                        errors.append(
                            f"{abs_file_path} was not parsed into AST (from {document_path})"
                        )
                        continue
                    if header and not contains_header(asts[abs_file_path], header):
                        errors.append(
                            f"{header} from {document_path} does not exist in {abs_file_path}"
                        )
                        continue
    if errors:
        raise ValueError(
            "Errors:\n" + "\n".join([f"- {error}" for error in errors]) + f"\n{len(errors)} errors"
        )


if __name__ == "__main__":
    main()
