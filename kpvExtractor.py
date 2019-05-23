#from tika import parser
#import PyPDF2

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.converter import TextConverter, PDFPageAggregator, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdevice import PDFDevice
import pdfminer
import re
import csv
import os
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
# from io import StringIO, BytesIO

# """
# Extract PDF text using PDFMiner. Adapted from
# http://stackoverflow.com/questions/5725278/python-help-using-pdfminer-as-a-library
# """
#
# def pdf_to_xml(pdfname):
#     # PDFMiner boilerplate
#     rsrcmgr = PDFResourceManager()
#     bio = BytesIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = XMLConverter(rsrcmgr, bio, codec=codec, laparams=laparams)
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     # Extract text
#     fp = open(pdfname, 'rb')
#     for page in PDFPage.get_pages(fp):
#         interpreter.process_page(page)
#     fp.close()
#     # Get text from StringIO
#     text = bio.getvalue()
#     # Cleanup
#     device.close()
#     bio.close()
#     return text
#
#
# def extract_text_from_pdf(pdf_path):
#     resource_manager = PDFResourceManager()
#     fake_file_handle = StringIO()
#     converter = TextConverter(resource_manager, fake_file_handle)
#     page_interpreter = PDFPageInterpreter(resource_manager, converter)
#
#     with open(pdf_path, 'rb') as fh:
#         for page in PDFPage.get_pages(fh,
#                                       caching=True,
#                                       check_extractable=True):
#             page_interpreter.process_page(page)
#
#         text = fake_file_handle.getvalue()
#
#     # close open handles
#     converter.close()
#     fake_file_handle.close()
#
#     if text:
#         return text

if __name__ == '__main__':
    """
    Ideas:
    -Use distance formula with the given coordinates to help identify important information
    -Organize data into 2d array of line, x coord, y coord, number of appearances
    -This code is so gross RIP
    """

    # raw = parser.from_file('USVisaForm.pdf')
    #print(raw['content'])
    #print('\n')
    with open('pdfData.csv', mode='w') as csv_file:
        fieldnames = ['Document', 'Address', 'Name', 'Email', 'Phone #', 'Social Sec', 'Time']
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if os.stat("pdfData.csv").st_size == 0:
            writer.writerow(fieldnames)

        def parse_document(pdfname):
            # Open a PDF file.writer
            fp = open(pdfname, 'rb')
            # Create a PDF parser object associated with the file object.
            parser = PDFParser(fp)
            # Create a PDF document object that stores the document structure.
            # Password for initialization as 2nd parameter
            document = PDFDocument(parser)
            # Check if the document allows text extraction. If not, abort.
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed
            # Create a PDF resource manager object that stores shared resources.
            rsrcmgr = PDFResourceManager()
            # Create a PDF device object.
            device = PDFDevice(rsrcmgr)
            # BEGIN LAYOUT ANALYSIS
            # Set parameters for analysis.
            laparams = LAParams()
            # Create a PDF page aggregator object.
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            # Create a PDF interpreter object.
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            def parse_obj(lt_objs):
                # loop over the object list
                # textList = []
                for obj in lt_objs:
                    # if it's a textbox, print text and location
                    if isinstance(obj, pdfminer.layout.LTTextLineHorizontal):
                        # print("%6d, %6d, %s" % (obj.bbox[0], obj.bbox[1], obj.get_text().replace('\n', ' _')))
                        important(obj.get_text().replace('\n', ' _'))
                        # textItem = {
                        #     'text': obj.get_text().replace('\n', '_'),
                        #     'count': 1
                        # }
                        # if (obj.get_text().replace('\n', '_')) not in textList:
                        #     textList.append(obj.get_text().replace('\n', '_'))
                        # else:
                        #     for item in textList:

                    # if it's a container, recurse
                    elif isinstance(obj, pdfminer.layout.LTFigure) or isinstance(obj, pdfminer.layout.LTTextBox):
                        parse_obj(obj._objs)

            # loop over all pages in the document
            for page in PDFPage.create_pages(document):
                # read the page into a layout object
                interpreter.process_page(page)
                layout = device.get_result()
                # extract text from this object
                parse_obj(layout._objs)

        def extract_phone_numbers(string):
            r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
            phone_numbers = r.findall(string)
            return [re.sub(r'\D', '', number) for number in phone_numbers]

        def extract_email_addresses(string):
            r = re.compile(r'[\w\.-]+@[\w\.-]+')
            return r.findall(string)

        def ie_preprocess(document):
            document = ' '.join([i for i in document.split() if i not in stop])
            sentences = nltk.sent_tokenize(document)
            sentences = [nltk.word_tokenize(sent) for sent in sentences]
            sentences = [nltk.pos_tag(sent) for sent in sentences]
            return sentences

        def extract_names(document):
            names = []
            sentences = ie_preprocess(document)
            for tagged_sentence in sentences:
                for chunk in nltk.ne_chunk(tagged_sentence):
                    if type(chunk) == nltk.tree.Tree:
                        if chunk.label() == 'PERSON':
                            names.append(' '.join([c[0] for c in chunk]))
            return names

        # fieldnames = ['Document', 'Address', 'Name', 'Email', 'Phone #', 'Social Sec', 'Time']
        def important(line):
            line = str(line.encode('utf-8'))
            line = line[2:-2]
            timeTags = ["day", " sun", " mon", " tue", " wed", " thu", " fri", " sat", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            for tag in timeTags:
                if tag in line.lower() and any(char.isdigit() for char in line):
                    print("Possible Date/Time Info: " + str(line))
                    writer.writerow([pdf, '', '', '', '', '', str(line)])
                    return
            if (re.search("\d{3}( |-)*\d{3}( |-)*\d{4}", line)):
                print("Possible Phone number: " + str(extract_phone_numbers(line)))
                for num in extract_phone_numbers(line):
                    writer.writerow([pdf, '', '', '', num, '', ''])
            if re.search("[0-9]{3}( |-)[0-9]{2}( |-)[0-9]{4}", line):
                print("Possible Social Sec Info:" + line)
                writer.writerow([pdf, '', '', '', '', str(line), ''])
            emailTags = [".com", ".edu", ".org", ".net"]
            for tag in emailTags:
                if tag in line.lower() and '@' in line:
                    print("Possible E-mail Info: " + str(extract_email_addresses(line)))
                    for email in extract_email_addresses(line):
                        writer.writerow([pdf, '', '', email, '', '', ''])
                    return
            nameTags = [" name", " mr", " ms", " mrs", " jr"]
            for tag in nameTags:
                if tag in line.lower() and len(str(extract_names(line))) > 2:
                    print("Possible Name Info: " + str(extract_names(line)))
                    for name in extract_names(line):
                        writer.writerow([pdf, '', name, '', '', '', ''])
            if len(str(extract_names(line))) > 2:
                print("Possible name Info: " + str(extract_names(line)))
                for name in extract_names(line):
                    writer.writerow([pdf, '', name, '', '', '', ''])
            addressTags = [" street ", " st ", " road ", " rd ", " circle ", " cl ", " lane ", " ln ", " city ", " north", " east", " south", " west", " land ", " location ", " address "]
            for tag in addressTags:
                if tag in line.lower():
                    print("Possible Address Info:" + line)
                    writer.writerow([pdf, str(line), '', '', '', '', ''])
                    return
            if re.search("[a-z]+, [a-z]{2} [0-9]+(-[0-9]+)*", line.lower()):
                print("Possible Address Info:" + line)
                writer.writerow([pdf, str(line), '', '', '', '', ''])
                return

        pdfList = ["1040.pdf", "20071040A.pdf", "ETicketReceipt.pdf", "NWCSampleInvoice.pdf", "PDFInvoiceSample.pdf", "QuotationSample.pdf", "TicketSample.pdf", "USVisaForm.pdf", "WellvibeQuotationForm.pdf"]
        for pdf in pdfList:
            print("Document: " + pdf + "\n")
            parse_document(pdf)
            print("\n")

    #pdf_to_xml('20071040A.pdf').replace('', '')

    #failed attempts
    # pdf_file = open('USVisaForm.pdf', 'rb')
    # read_pdf = PyPDF2.PdfFileReader(pdf_file)
    # number_of_pages = read_pdf.getNumPages()
    # page = read_pdf.getPage(0)
    # page_content = page.extractText()
    # print(page_content.encode('utf-8'))
    # pdf_file2 = open('USVisaForm.pdf', 'rb')
    # print(read_pdf.getFields())
    # print(read_pdf.getDocumentInfo())

    #print(extract_text_from_pdf('20071040A.pdf'))