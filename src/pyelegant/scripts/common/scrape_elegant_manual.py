import argparse
from datetime import datetime
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests
from ruamel import yaml


def _scrape_elem_prop_tables(url):

    # Send an HTTP request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        for iElem, element in enumerate(soup.find_all("h4")):
            page_title = element.get_text()
        assert iElem == 0

        dash_like_char = chr(8212)
        section_number, elem_type, description = re.match(
            f"(\d+\.\d+)\s+(\w+){dash_like_char}(.+)", page_title
        ).groups()
        page_title = dict(
            section_number=section_number, elem_type=elem_type, description=description
        )

        tables = soup.find_all("table")

        # Iterate through tables
        table_lines = []
        for table in tables:
            # Find all rows in the table
            rows = table.find_all("tr")

            # Iterate through rows
            lines = []
            for row in rows:
                # Find all cells in the row
                cells = row.find_all(["td", "th"])

                # Extract and print the content of each cell
                row_tokens = []
                for iCol, cell in enumerate(cells):
                    cell_str = cell.text.strip()
                    # print(cell_str, end='\t')
                    if iCol == 0 and cell_str == "":
                        break
                    cell_str = " ".join(cell_str.split())
                    row_tokens.append(cell_str)

                if row_tokens != []:
                    # lines.append(' ### '.join(row_tokens))
                    lines.append(row_tokens)

            # table_lines.append('\n'.join(lines))
            if lines != []:
                table_lines.append(lines)

    return dict(url=url, title=page_title, table_lines=table_lines)


def scrape_elem_dict():

    tStart = f"{datetime.now():%Y-%m-%d %H:%M:%S}"

    top_url = (
        "https://ops.aps.anl.gov/manuals/elegant_latest/elegantse10.html#x121-12000010"
    )

    # Send an HTTP request to the URL
    response = requests.get(top_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links within <span class="subsectionToc">
        links = soup.select("span.subsectionToc > a[href]")
        n_links = len(links)

        all_d = {}

        # Follow each link and recursively scrape its content
        for iLink, link in enumerate(links):
            print(f"Parsing Link #{iLink+1}/{n_links}")
            next_url = urljoin(top_url, link["href"])
            page_d = _scrape_elem_prop_tables(next_url)

            if False:
                # Save the extracted text to the output file
                table_text = "\n----------------\n".join(page_d["table_lines"])
                with open(output_file, "a", encoding="utf-8") as file:
                    file.write(
                        f"### Scraped from {next_url}:\nTitle: {page_d['title']}\n\n{table_text}\n\n"
                    )
            else:
                if False:
                    all_d[next_url] = page_d
                else:
                    elem_type = page_d["title"]["elem_type"]
                    del page_d["title"]["elem_type"]
                    all_d[elem_type] = page_d

            if False:  # enable this section for debugging
                if iLink >= 2:
                    break

    else:
        print(
            f"Failed to retrieve the page {top_url}. Status code: {response.status_code}"
        )

    col_labels = ["Parameter Name", "Units", "Type", "Default", "Description"]

    if False:
        table_lines = all_d["CSBEND"]["table_lines"]

        merged_table = [col_labels]

        for lines in table_lines:
            assert lines[0] == col_labels
            for line in lines[1:]:
                merged_table.append(line)

        print("\n".join([" # ".join(tokens) for tokens in merged_table]))

    for elem_type, sub_d in all_d.items():
        merged_table = [get_yaml_flow_style_list(col_labels)]
        for lines in sub_d["table_lines"]:
            if lines[0] != col_labels:
                continue
            for line in lines[1:]:
                line = get_yaml_flow_style_list(line)
                merged_table.append(line)

        sub_d["table"] = merged_table
        del sub_d["table_lines"]

    tEnd = f"{datetime.now():%Y-%m-%d %H:%M:%S}"

    out = dict(timestamp_start=tStart, timestamp_end=tEnd, elements=all_d)

    y = yaml.YAML()
    y.preserve_quotes = True
    y.width = 110
    y.boolean_representation = ["False", "True"]
    y.indent(
        mapping=2, sequence=2, offset=0
    )  # Default: (mapping=2, sequence=2, offset=0)

    with open("../../elegant_elem_dict.yaml", "w") as f:
        y.dump(out, f)


def get_yaml_flow_style_list(L):

    L = yaml.comments.CommentedSeq(L)
    L.fa.set_flow_style()

    return L


def get_parsed_args():
    """"""

    parser = argparse.ArgumentParser(
        prog="pyele_scrape_manual", description="Scraper for ELEGANT manual web pages"
    )
    parser.add_argument(
        "manual_section",
        type=str,
        help="""\
    ELEGANT manual section name to be scraped.""",
    )

    args = parser.parse_args()
    if False:
        print(f"Manual Section = {args.manual_section}")

    return args


def main():
    """"""

    args = get_parsed_args()

    if args.manual_section == "elem_dict":
        scrape_elem_dict()
    else:
        raise ValueError(args)


if __name__ == "__main__":

    """
    Occasionally run this script with the argument "elem_dict" in order to update
    "../../elegant_elem_dict.yaml"
    """
    main()
