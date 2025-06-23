import stanza
import sys

nlp = stanza.Pipeline('en')
input_file = sys.argv[1]
if ".txt" in input_file :
    output_file = input_file.replace(".txt", "_tagged.txt")
else :
    output_file = input_file + "_tagged.txt"

sentences = []

with open(input_file, "r", encoding = "utf-8") as f :
    for line in f.readlines() :
        line = line.strip()
        sentences.append(line)

with open(output_file, "w", encoding = "utf-8") as f :
    for i, sentence in enumerate(sentences) :
        print(sentence)
        print("<s id=\"%d\">"%(i+1), file = f)
        doc = nlp(sentence)

        for sent in doc.sentences :
            for word in sent.words :
                print(word.text, word.upos, word.lemma, word.id, word.head, sent.words[word.head-1].text if word.head > 0 else "root", word.deprel, sep = "\t", file = f)
            print("</s>", file = f)


