import os
import pandas as pd
import configparser
import torch
import dropbox
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline, PromptTemplate
import re
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextStreamer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from langchain.chains import RetrievalQA
import pickle

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."


class ConfigManager:
    @staticmethod
    def get_token(file_path):
        config = configparser.ConfigParser()
        config.read(file_path)
        return config['DEFAULT']['dropbox_token']


class EncodingOverride:
    @staticmethod
    def set_preferred_encoding():
        import locale
        locale.getpreferredencoding = lambda: "UTF-8"


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages = []

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        self.pages = loader.load_and_split()
        return self.pages


class ChainGenerator:
    def __init__(self, device, system_prompt, pages):
        self.device = device
        self.system_prompt = system_prompt
        self.pages = pages
        self.template = self.generate_prompt("""
                                                    {context}

                                                    Question: {question}
                                                    """, self.system_prompt)

    def generate_prompt(self, prompt, system_prompt=SYSTEM_PROMPT):
        return f"""
    [INST] <>
    {system_prompt}
    <>

    {prompt} [/INST]
    """.strip()

    def create_chain(self):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",
                                                   model_kwargs={"device": self.device})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64,
                                                       separators=["\n\n", "\n", "(?<=\. )", " ", ""])
        texts = text_splitter.split_documents(self.pages)

        db = Chroma.from_documents(texts, embeddings)

        model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            revision="gptq-4bit-128g-actorder_True",
            model_basename="model",
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False,
            device=self.device,
            quantize_config=None,
        )

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=True
        )

        return qa_chain


class ExtractionTemplate:
    # Define static format instructions for different extraction types
    FORMAT_INSTRUCTIONS_INT = """your response should only be an integer and start with "```" and end with "```" """
    FORMAT_INSTRUCTIONS_BOOL = """your response should only be 'True' or 'False' and start with "```" and end with "```" """
    FORMAT_INSTRUCTIONS_STR = """your response should start with one set of "```" and end with "```" """

    # Define static templates for different extraction types
    TEMPLATE_INT = """\
        Given this question and answer, can you extract the {attribute} as an integer?  Please respond in a concise manner, if there is no information, say '-1'.
        question: {question}
        
        answer: {answer}
        
        format: {format_instructions}
        """

    TEMPLATE_BOOL = """\
        Given this question and answer, can you answer only 'True' or 'False' for {attribute}?  Please respond in a concise manner, if there is no information, say 'False'.
        question: {question}
        
        answer: {answer}
        
        format: {format_instructions}
        """

    TEMPLATE_STR = """\
        Given this answer, can you put the answer into the following format? If based on the answer no information can be found, respond 'None'. Respond in a concise manner, do not use first person.
        
        {answer}
        
        format: {format_instructions}
        """


class DataExtractor:
    def __init__(self, chain):
        self.chain = chain

    def extract_from_file(self, extract_ques_df):
        row_dict = {}
        for i in range(len(extract_ques_df)):
            cur_attr = extract_ques_df.iloc[i]['attr']
            cur_ques = extract_ques_df.iloc[i]['ques']
            cur_extract_type = extract_ques_df.iloc[i]['extract_type']
            result = self.chain(cur_ques)

            # Select the appropriate template and format instructions based on cur_extract_type
            if cur_extract_type == 'int':
                selected_template = ExtractionTemplate.TEMPLATE_INT
                selected_format_instructions = ExtractionTemplate.FORMAT_INSTRUCTIONS_INT
            elif cur_extract_type == 'boolean':
                selected_template = ExtractionTemplate.TEMPLATE_BOOL
                selected_format_instructions = ExtractionTemplate.FORMAT_INSTRUCTIONS_BOOL
            else:  # Default case
                selected_template = ExtractionTemplate.TEMPLATE_STR
                selected_format_instructions = ExtractionTemplate.FORMAT_INSTRUCTIONS_STR

            # Create the ChatPromptTemplate instance with the selected template
            prompt = ChatPromptTemplate.from_template(template=selected_template)

            # Format the messages using the selected format instructions
            messages = prompt.format_messages(attribute=cur_attr, question=cur_ques, answer=result['result'].strip(),
                                              format_instructions=selected_format_instructions)

            response = self.chain(messages[0].content)
            ans_str = re.search(r'```(.*?)```', response['result'], re.DOTALL)
            if ans_str:
                ans_substr = ans_str.group(1).strip()
                print(ans_substr)
            else:
                ans_substr = np.nan
                print("No answer substring found")

            row_dict[cur_attr] = ans_substr

        return row_dict


class FileManager:
    @staticmethod
    def ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")

    @staticmethod
    def read_csv(file_path):
        return pd.read_csv(file_path, encoding='utf-8')

    @staticmethod
    def save_to_pickle(row_dict, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(row_dict, f)
        return

    @staticmethod
    def upload_to_dropbox(file_path, savename, dropbox_token):
        dbx = dropbox.Dropbox(dropbox_token)
        with open(file_path, 'rb') as f:
            dbx.files_upload(f.read(), f'/{savename}.pickle')
        return

    @staticmethod
    def save_to_csv(row_list, file_path):
        df = pd.DataFrame(row_list)
        df.to_csv(file_path)
        return

class MainProcess:
    def __init__(self, config_path, base_path, output_path, file_type='10-K'):
        self.dropbox_token = ConfigManager.get_token(config_path)
        self.base_path = base_path
        self.output_path = output_path
        self.file_type = file_type
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


    def run(self):
        EncodingOverride.set_preferred_encoding()

        # Main process logic, looping over tickers and PDFs
        extract_ques_df = FileManager.read_csv(f"ques_to_ask_{self.file_type}.csv")
        ticker_sample = os.listdir(self.base_path)

        for ticker in ticker_sample:
            cur_ticker_row_list = []  # Incremental save for each ticker
            pdf_list = [x for x in os.listdir(f'{self.base_path}/{ticker}/{self.file_type}') if x.endswith('.pdf')]
            for pdf_fn in pdf_list:
                pdf_processor = PDFProcessor(f'{self.base_path}/{ticker}/{self.file_type}/{pdf_fn}')
                pages = pdf_processor.load_pdf()

                chain_generator = ChainGenerator(self.device, SYSTEM_PROMPT, pages)
                qa_chain = chain_generator.create_chain()

                data_extractor = DataExtractor(qa_chain)

                row_dict = data_extractor.extract_from_file(extract_ques_df)

                row_dict['sec_file_name'] = pdf_fn

                cur_ticker_row_list.append(row_dict)

                FileManager.ensure_folder_exists(f"{self.output_path}/{ticker}")
                savename = pdf_fn.split('.pdf')[0]

                # Incrementally save each row as pickle file (in case of out of memory crashes, won't lose entire run)
                FileManager.save_to_pickle(row_dict, f"{self.output_path}/{ticker}/{savename}.pickle")
                FileManager.upload_to_dropbox(f"{self.output_path}/{ticker}/{savename}.pickle", savename, self.dropbox_token)

                torch.cuda.empty_cache()

            FileManager.save_to_csv(cur_ticker_row_list, f'{self.output_path}/{ticker}.csv')

        FileManager.save_to_csv(cur_ticker_row_list, f'{self.output_path}/all_tickers.csv')


# Usage example
if __name__ == "__main__":
    process = MainProcess(r'/config.ini', r'/pdfs',
                          r'/llm_output','10-K')
    process.run()
