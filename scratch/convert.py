import docx

def md_to_docx(md_path, docx_path):
    doc = docx.Document()
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        doc.add_paragraph(line.strip())
    doc.save(docx_path)
    print("Saved docx!")

if __name__ == '__main__':
    md_to_docx('report.md', 'report.docx')
