import os
from datetime import date


class SimplePDF:
    def __init__(self):
        self.objects = []  # list[bytes]

    def add_object(self, body: bytes) -> int:
        self.objects.append(body)
        return len(self.objects)  # 1-based object number

    @staticmethod
    def _escape_text(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    def make_page_with_lines(self, lines, font_size=12, left=72, top=720, leading=18):
        # Font object (Helvetica)
        font_obj_num = self.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        # Content stream
        content_lines = [
            "BT",
            f"/F1 {font_size} Tf",
            f"{left} {top} Td",
        ]
        for i, line in enumerate(lines):
            txt = self._escape_text(line)
            if i > 0:
                content_lines.append(f"0 -{leading} Td")
            content_lines.append(f"({txt}) Tj")
        content_lines.append("ET")
        content_text = "\n".join(content_lines) + "\n"
        content_bytes = content_text.encode("utf-8")

        content_stream = b"<< /Length " + str(len(content_bytes)).encode() + b" >>\nstream\n" + content_bytes + b"endstream"
        contents_obj_num = self.add_object(content_stream)

        # Page object
        page_obj_num = self.add_object(
            (
                b"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
                b"/Contents " + f"{contents_obj_num} 0 R".encode() + b" "
                b"/Resources << /Font << /F1 " + f"{font_obj_num} 0 R".encode() + b" >> >> >>"
            )
        )

        return page_obj_num

    def build(self, page_obj_nums):
        # Pages tree
        kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
        pages_dict = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>".encode()
        pages_obj_num = self.add_object(pages_dict)

        # Fix each page's Parent reference to this pages obj
        fixed_objects = []
        for idx, raw in enumerate(self.objects, start=1):
            if raw.startswith(b"<< /Type /Page") and b"/Parent 0 0 R" in raw:
                fixed = raw.replace(b"/Parent 0 0 R", f"/Parent {pages_obj_num} 0 R".encode())
                fixed_objects.append(fixed)
            else:
                fixed_objects.append(raw)
        self.objects = fixed_objects

        # Catalog
        catalog_obj_num = self.add_object(f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode())

        # Write final PDF bytes with xref
        out = bytearray()
        out += b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"

        offsets = [0]  # obj 0 is the free object
        for i, body in enumerate(self.objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode()
            out += body + b"\nendobj\n"

        xref_start = len(out)
        out += f"xref\n0 {len(self.objects)+1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()

        out += b"trailer\n"
        out += (
            b"<< "
            + f"/Size {len(self.objects)+1} ".encode()
            + f"/Root {catalog_obj_num} 0 R ".encode()
            + b">>\n"
        )
        out += b"startxref\n" + str(xref_start).encode() + b"\n%%EOF\n"
        return bytes(out)


def write_pdf(path: str, lines):
    pdf = SimplePDF()
    page = pdf.make_page_with_lines(lines)
    data = pdf.build([page])
    with open(path, "wb") as f:
        f.write(data)


def main(output_dir: str = "data/in"):
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    datasets = {
        "pii_basic.pdf": [
            "Employee Record",
            f"Date: {today}",
            "Name: John A. Doe",
            "SSN: 123-45-6789",
            "Email: john.doe@example.com",
            "Phone: (415) 555-0123",
            "Address: 1234 Market St, Apt 5B, San Francisco, CA 94103",
        ],
        "pii_mixed.pdf": [
            "Client Intake Form",
            f"Received: {today}",
            "Full Name: María-José Carreño Quiñones",
            "DOB: 1989-07-15",
            "Driver License: D123-456-789-123",
            "Email: maria.q@example.co.uk",
            "Alt Email: m-c.quinones+admin@sample.org",
            "Mobile: +1 202-555-0188",
            "IP: 203.0.113.42",
        ],
        "pii_invoice.pdf": [
            "Invoice #INV-10023",
            f"Date: {today}",
            "Bill To: Alex Johnson",
            "Card: 4111 1111 1111 1111",
            "Billing Email: alex.johnson@contoso.com",
            "Billing Phone: 212-555-0199",
            "Billing Address: 77 Broadway, New York, NY 10006",
            "Amount Due: $1,245.77",
        ],
        "pii_resume.pdf": [
            "Curriculum Vitae - Priya Sharma",
            "Email: priya.sharma+jobs@gmail.com",
            "Phone: 650.555.0007",
            "LinkedIn: linkedin.com/in/priyasharma",
            "Address: 88 King St, San Mateo, CA 94401",
        ],
        "pii_form.pdf": [
            "Healthcare Registration",
            "Patient: Robert O'Connor",
            "Medical Record #: MRN0098123",
            "Insurance Member ID: XHJ-556-21-9921",
            "Emergency Contact: Sarah O'Connor (415) 555-7788",
            "Email: sarah.oconnor@example.net",
        ],
        "pii_varied.pdf": [
            "Support Ticket",
            "Requester: Chen Wei",
            "Account #: 987654321",
            "Email: c.wei@company.cn",
            "Alt Phone: +44 20 7946 0958",
            "Address: 221B Baker Street, London NW1 6XE, UK",
        ],
    }

    for filename, lines in datasets.items():
        path = os.path.join(output_dir, filename)
        write_pdf(path, lines)


if __name__ == "__main__":
    main()

