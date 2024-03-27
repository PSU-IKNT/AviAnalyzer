FROM python

RUN mkdir -p /Analyzer
WORKDIR /Analyzer
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD DataFrame.py .
CMD python ./DataFrame.py