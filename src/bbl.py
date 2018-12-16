import urllib.request
import urllib

def extract(html):
    content = ""
    for l in html.splitlines():
        if 'class="verse"' in l:
            for c in l:
                if c==' ' or (c >= u"\U00001200" and c <= u"\U0000135A"):
                    content += c
    content = ' '.join(content.split())
    return content


def save(book, chapter):
    response = get(book, chapter)
    if response[0]:
        base = "./data/bbl/%02d-%02d" % (book,chapter)
        with open(base, "w") as f:
            f.write(response[1])
        url = FILE_BASE + "%d/%d.mp3" % (book, chapter)
        urllib.request.urlretrieve (url, base+'.mp3')
    return response[0]


def get(book, chapter):
    b = "%02d"%book
    url = WEB_BASE + b + "/" + str(chapter) + ".htm"
    request = urllib.request.Request(url,headers={'User-Agent': USER_AGENT})
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode(response.headers.get_content_charset())
        if "404 Not Found" in html:
            return (False, "")
        else:
            extracted = extract(html)
            return (True, extracted)
    except:
        return (False, "")

def get_all():
    book = 1
    chapter = 1
    while(True):
        chapter = 1
        if not get(book, chapter)[0]:
            print("Finished books")
            break
        while(True):
            print("Requesting book %d chapter %d" % (book, chapter))
            exists = save(book, chapter)
            if not exists:
                print("Finished chapters for book %d" % book)
                break
            chapter+=1
        book+=1
