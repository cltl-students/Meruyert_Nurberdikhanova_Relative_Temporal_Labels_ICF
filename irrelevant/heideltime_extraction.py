from python_heideltime import Heideltime

parser = Heideltime()
parser.set_document_type('NEWS')
parser.set_language('DUTCH')
print(parser.parse('Gisteren kocht ik een kat!'))
