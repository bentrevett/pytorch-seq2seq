FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate torch-seq2seq" > ~/.bashrc

ENV PATH /opt/conda/envs/torch-seq2seq/bin:$PATH

RUN python -m spacy download en \
  && python -m spacy download de

WORKDIR home

EXPOSE 8888
