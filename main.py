import PyPDF2
def word_count_in_pdf(path):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        list_text = [page.extract_text() for page in reader.pages]
        text = "".join(list_text)
        return len(text.split())

word_count_in_pdf("data/1709219978_240301_142017.pdf")

if __name__ == "__main__":
    print(word_count_in_pdf("data/1709219978_240301_142017.pdf"))