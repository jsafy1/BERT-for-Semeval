"""
This script transforms the Semeval Task 5: Hyperpartisan News Detection data
provided in XML format, to CSV format for easier use.
"""
import pandas as pd
import xml.etree.cElementTree as et
import numpy as np

gfiles = ["./ground-truth-training-byarticle-20181122.xml",
          "./ground-truth-training-bypublisher-20181122.xml",
          "./ground-truth-validation-bypublisher-20181122.xml"]

gdfCols = ["hyperpartisan", "id", "labeled-by", "url", "bias"]

files = ["./articles-training-byarticle-20181122.xml",
         "./articles-training-bypublisher-20181122.xml",
         "./articles-validation-bypublisher-20181122.xml"]

dfCols = ["id", "published-at", "title", "text"]

# Handles articles
for _file in files:
    df = pd.DataFrame(columns=dfCols)
    index = 1
    i = []
    p = []
    ti = []
    te = []
    for node in et.parse(_file).getroot():
        i.append(node.attrib.get("id"))
        p.append(node.attrib.get("published-at"))
        ti.append(node.attrib.get("title"))
        if node.text is not None:
            node.text = None
        article = ""
        for paragraph in node.itertext():
            article += paragraph
        te.append(article)
        index += 1
        if index % 100 == 0:
            print(index)
    df["id"], df["published-at"], df["title"], df["text"] = pd.Series(i), pd.Series(p), pd.Series(ti), pd.Series(te)
    print(df.shape)
    df.to_csv(_file[:-4] + ".csv")

# Handles ground truth
for gfile in gfiles:
    df = pd.DataFrame(columns=gdfCols)
    index = 1
    h = []
    i = []
    l = []
    u = []
    b = []
    for node in et.parse(gfile).getroot():
        h.append(node.attrib.get("hyperpartisan"))
        i.append(node.attrib.get("id"))
        l.append(node.attrib.get("labeled-by"))
        u.append(node.attrib.get("url"))
        b.append(node.attrib.get("bias") if node.attrib.get("bias") is not None else "")
        index += 1
        if index % 100 == 0:
            print(index)
    df["hyperpartisan"], df["id"], df["labeled-by"], df["url"], df["bias"] = pd.Series(h), pd.Series(i), pd.Series(l), pd.Series(u), pd.Series(b)
    print(df.shape)
    df.to_csv(gfile[:-4] + ".csv")

# Merges
for i in range(len(files)):
    print(i)
    df_file, df_gfile = pd.read_csv(files[i][:-4] + ".csv", sep=','), pd.read_csv(gfiles[i][:-4] + ".csv", sep=',')
    df = pd.merge(df_file, df_gfile, on="id")
    print("Done merging")
    df = df[df.columns.difference(["", "Unnamed: 0_x", "Unnamed: 0_y", "Unnamed"])]
    df["hyperpartisan"] = np.array([0 if val == False else 1 for val in df["hyperpartisan"]])
    df.to_csv(gfiles[i][:-4] + "_merged.csv")
    print("Done saving")
