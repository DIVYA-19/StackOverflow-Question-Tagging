import json
import requests
import streamlit as st
import streamlit.components.v1 as components


st.title('Question tagger')

st.text('try it yourself!!!!')

# api url
post_url = "http://localhost:8000/get_tags_single"

html= '''<html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            </head>
            <body>'''

title = st.text_input("title")
body = st.text_area("body")
if st.button("Done"):

    data = json.dumps({'title': title, 'body': body})
    # returns predicted tags in format - {'0':[tag1, tag2,..]}
    tags = requests.post(post_url, data=data)
    tags = tags.json()["0"]

    for i in range(len(tags)):
        html+= '<span style="margin-right: 10px;padding: 8px; font-size: 16px;" class="badge badge-success">'
        html += tags[i]
        print(tags[i])
        html+= "</span>"
    html += "</body></html>"

components.html(html)
