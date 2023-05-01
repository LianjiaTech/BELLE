import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    "-d",
    help="path to articles xml",
    default="enwiki-20230401-pages-articles-multistream.xml",
)

parser.add_argument(
    "--output",
    "-o",
    help="path to extracted urls file",
    default="./extracted_urls.txt",
)

parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="show progress",
)

args = parser.parse_args()


def get_urls():
    with open(args.data, "r", errors="ignore") as f, open(args.output, "w") as out:
        for i, line in enumerate(f, start=1):
            refs = re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line)
            if refs is not None:
                results = re.findall(
                    r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))",
                    refs.group(0),
                )
                if len(results) > 0:
                    for result in results:
                        out.write(result + "\n")

            if args.verbose and i % 1000000 == 0:
                print("Lines searched: {}".format(i), end="\r")


def main():
    get_urls()


if __name__ == "__main__":
    main()
