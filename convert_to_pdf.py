from markdown_pdf import MarkdownPdf, Section
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: .venv/bin/python convert.py <input.md> <output.pdf>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md_content))
    pdf.save(output_file)
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    main()