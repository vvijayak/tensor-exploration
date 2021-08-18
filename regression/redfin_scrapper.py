from bs4 import BeautifulSoup
import requests
import urllib.request


DATA_FILE_PATH = '../data/real_estate/'
DOMAIN = 'https://www.redfin.com'
TIME_FRAME = '1yr'
ZIP_CODE_FILE = 'bay_area_zip_codes.txt'


def download_csv(url, zip_code):

    # get data frame from url
    if 'sold_within' in url:
        data_type = 'train'
    else:
        data_type = 'test'

    file_name = DATA_FILE_PATH + f'{zip_code}-{TIME_FRAME}-{data_type}.csv'

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    f = urllib.request.urlretrieve(url, file_name)

    print(f'files saved: {file_name}')


def get_download_link(zip_code):
    """
    Using bs4 get download links for CSVs from redfin
    :param zip_code: 5 digit city zip-code
    :return: download link for zip-code
    """

    search_url = DOMAIN+f'/zipcode/{zip_code}/filter/include=sold-{TIME_FRAME}/'

    # get bs4 parsed object
    header = {'User-agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=header)
    soup = BeautifulSoup(response.text, "html.parser")

    # clean up link
    download_link = str(soup.findAll("a", {"id": "download-and-save"})).replace(
        '[<a class="downloadLink" href="','').replace\
        ('" id="download-and-save" target="_self">(Download All)</a>]', '').replace('amp;', '')
    download_link = DOMAIN + download_link
    print(f'download link scraped:{download_link}')

    return download_link


def main():

    failures = []
    with open(ZIP_CODE_FILE) as f:
        zip_codes = f.read().splitlines()

    for zip_code in zip_codes:
        try:
            download_csv(get_download_link(zip_code), zip_code)
        except Exception:
            failures.append(zip_code)

    print(failures)


if __name__ == '__main__':
    main()
