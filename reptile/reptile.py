import requests
from lxml import etree


def get_selector_content(url,headers_update=None,selector=None):
    '''
    请求url，并通过xpath提取selector_xpath对应内容
    '''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
    }
    if headers_update:
        headers.update(headers_update)

    response = requests.get(url, headers=headers)
    html_str = response.text

    # if 'https://so.gushiwen.cn/nocdn/ajaxfanyi.aspx?' in url:
    #     print(html_str)

    tree = etree.HTML(html_str)

    if selector == 'json':
        return response.json()
    elif selector == 'text':
        return response.text
    elif selector == 'content':
        return response.content
    elif selector:
        return tree.xpath(selector)
    else:
        return tree


if __name__ == '__main__':
    page = 0
    for a in range(7):
        url = f"https://image.baidu.com/search/acjson?tn=resultjson_com&logid=11180288326636287564&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E6%B8%85%E6%B4%81%E6%9C%BA%E5%99%A8%E4%BA%BA&queryWord=%E6%B8%85%E6%B4%81%E6%9C%BA%E5%99%A8%E4%BA%BA&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn={page}&rn=30&gsm=78&1652074678117="
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        html_test = response.json()
        image_url_list = html_test.get('data')
        for i in image_url_list:
            if len(i)==0:
                continue
            image_url = i.get('middleURL')
            image_name = i.get('fromPageTitleEnc')
            try:
                file_name = './images/' + image_name + '.jpg'
                with open(file_name, 'wb') as fp:
                    fp.write(get_selector_content(image_url, selector='content'))
                print(file_name, '下载完成！')
            except Exception:
                pass
        page = page + 30