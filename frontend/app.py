import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sms import main as sms_main
from jobs import main as job_main
from voicemessage import main as audio_main

# sidebar
with st.sidebar:
    selected = option_menu('Spam Detection', [
        'SMS',
        'EMAIL',
        'Voice Messages',
        'Twitter Comments',
        'Youtube Comments',
        'Review Of Product',
        'Links',
        'News',
        'Jobs',
        'Dashboard'
    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)



# Diabetes prediction page
if selected == 'SMS':
    sms_main()  
if selected == 'Jobs':
    job_main() 
if selected == 'Voice Messages':
    audio_main() 
    

