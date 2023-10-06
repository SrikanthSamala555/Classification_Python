import utils
import streamlit as st
import requests
import time
import os
import json
import sqldata
import PromptsConstants
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool,ZeroShotAgent,AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import Sequence
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import langchain
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import create_extraction_chain
from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
# from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
langchain.debug=True
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool, StructuredTool, Tool, tool

os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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"
st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with Classification API')
st.write('Equipped with internet access and classification API access, enables users to ask questions about recent events')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/3_%F0%9F%8C%90_chatbot_with_internet_access.py)')


# embeddings = OpenAIEmbeddings(openai_api_key="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")
# loader = CSVLoader(file_path="./RevCodes.csv", encoding="utf-8")
# pages = loader.load()
# faiss_db = FAISS.from_documents(pages, embeddings)
# if os.path.exists("rev_store"):
#     local_db = FAISS.load_local("rev_store",embeddings)
#     local_db.merge_from(faiss_db)
#     print("Merge completed")
#     local_db.save_local("rev_store")
#     print("Updated index saved")
# else:
#     faiss_db.save_local(folder_path="rev_store")
#     print("New store created...")



class Classification(BaseModel):
       application:str=Field(description="default value IAConsumer")
       claimID:int=Field(description="default value 1234")
       payerClaimNumber:str=Field(description="Payor Claim Number")
       payerID:int=Field(description="default value 999")
       tradingPartnerID:int=Field(description="default value 999")
       effectiveDate:str=Field(description="default value 2021-06-18T14:16:29.896Z")
       tin:str=Field(description="default value 999")
       IsPayerFlaggedSupported:bool=Field(description="default value false")
       IsPayerFlaggedClaim:bool=Field(description="default value false")
       claimType:str=Field(description="default value ub")
       cptCodes:List[str]=Field(description="default value []")
       revCodes:List[str]=Field(description="default value []")
       posCodes:List[str]=Field(description="default value []")
       modifiers:List[str]=Field(description="default value []")
schema = {
    "properties":  {
        "application": {"type": "string"},
        "claimID": {"type":"integer"},
        "payerClaimNumber": {"type": "string"},
        "payerID": {"type":"integer"},
        "tradingPartnerID": {"type":"integer"},
        "effectiveDate": {"type": "string"},
        "tin": {"type": "string"},
        "IsPayerFlaggedSupported": {"type": "boolean"},
        "IsPayerFlaggedClaim": {"type": "boolean"},
        "claimType": {"type": "string"},
        "cptCodes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },
        "revCodes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    
                },
        "posCodes":{
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },
        "modifiers": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                }
    } ,   
     "required": ["application", "claimID","payerClaimNumber","payerID","tradingPartnerID","effectiveDate","tin","IsPayerFlaggedSupported", "IsPayerFlaggedClaim","claimType","cptCodes","posCodes","revCodes","modifiers"],
}
        
class ToolsDefination:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
    
    def csv_agent(store,input):
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.load_local(store, embeddings)
        results_with_scores_sim = vectors.similarity_search_with_score(input,k=10)
        results_with_scores_mmr = vectors.max_marginal_relevance_search(input,k=10)
        result=" Answer the question based on the context below. If you didn't find any exact match show possible matches " 
        index=0
        # is_under_score= False
        for doc, score in results_with_scores_sim:
            if index==0:
                result = result+ " Context: " +str(doc.page_content) +"\n"+results_with_scores_mmr[index].page_content
            else:
                result = result+ "\n"+str(doc.page_content) +"\n"+results_with_scores_mmr[index].page_content
            
        result=result + "\n Question: {Question} \n "
                
        result=result + "\n Answer with Code and Description with exact match :  "

        result=result + "\n Answer with Codes and Description with some best possible matches :  "

        print(result)
        PROMPT = PromptTemplate(
                    template=result, input_variables=["Question"]
                )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",max_retries=1, temperature=0.0)
                
        llm_chain = LLMChain(llm=llm,prompt=PROMPT)
        return llm_chain
    def execute_cpt_csv(input):   
        print("----------------------------1entered") 
        # time.sleep(30)  
        chain = ToolsDefination.csv_agent("index_Store",input)
        result = chain.run(input)
        print(result)
        return result
    def execute_pos_csv(input):   
        print("----------------------------2entered") 
        # time.sleep(30)  
        chain = ToolsDefination.csv_agent("pos_store",input)
        result = chain.run(input)
        print(result)
        return result
    def execute_rev_csv(input):    
        print("----------------------------3entered")
        # time.sleep(30)  
        chain = ToolsDefination.csv_agent("rev_store",input)
        result = chain.run(input)
        print(result)            
        return result
    
    def get_descriptions(output,type):  
        desc=""
        formatted_values=""
        try:
            dumpsJson=""
            dumpsJson =  json.loads(output)  
            values=(dumpsJson[type]) 
            if type=="cptCodes" and  len(values) > 0:           
                print("---fromatted values")
                formatted_values = ', '.join([f"'{value}'" for value in values])
                print(formatted_values)
                desc="CPTCodes:"+formatted_values
            elif (type=="posCodes" or type=="revCodes") and  len(values) > 0:
              print("---fromatted values")
              formatted_values = ', '.join([f"{value}" for value in values])
              print(formatted_values)
            desc=formatted_values
            connection = sqldata.sqlconnection()

            if type=="cptCodes" and  len(values) > 0:
             desc =connection.fetch_cpt_desc(formatted_values)
            elif type=="posCodes" and  len(values) > 0:
              desc =connection.fetch_pos_desc(formatted_values)  
            elif type=="revCodes" and  len(values) > 0:  
              desc==connection.fetch_rev_desc(formatted_values)  
            
            if desc=="":
                desc=formatted_values
            print (desc)
               
        except Exception as e:
            # Handling the exception and printing the error message
            print("An exception occurred:", str(e))
        return desc
    

    def _get_extraction_function(entity_schema: dict) -> dict:
        return {
            "name": "information_extraction",
            "description": "Extracts the relevant information from the passage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {"type": "array", "items": _convert_schema(entity_schema)}
                },
                "required": ["info"],
            },
        }
    def create_extraction_chain(
        schema: dict,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        verbose: bool = False,
    ) -> Chain:
            """Creates a chain that extracts information from a passage.

            Args:
                schema: The schema of the entities to extract.
                llm: The language model to use.
                prompt: The prompt to use for extraction.
                verbose: Whether to run in verbose mode. In verbose mode, some intermediate
                    logs will be printed to the console. Defaults to `langchain.verbose` value.

            Returns:
                Chain that can be used to extract information from a passage.
            """
            _EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned
        in the following passage together with their properties.
        
      
        Extract the properties mentioned in the 'information_extraction' function.

        
        Passage:
        {input}
        
       If a property is not present and is not required in the function parameters, do not include it in the output.
      
        
       
        
       

        """  # noqa: E501
            function = ToolsDefination._get_extraction_function(schema)
            extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
            output_parser = JsonKeyOutputFunctionsParser(key_name="info")
            llm_kwargs = get_llm_kwargs(function)
            chain = LLMChain(
                llm=llm,
                prompt=extraction_prompt,
                llm_kwargs=llm_kwargs,
                output_parser=output_parser,
                verbose=verbose,
            )
            return chain

    def NSAClassification(input):
        
        model_name = "text-davinci-003"
        temperature = 0.0
        model = OpenAI(model_name=model_name, temperature=temperature,max_retries=1,max_tokens=750)
        
        parser = PydanticOutputParser(pydantic_object=Classification)

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}.  Don't auto generate any value, Use default values only if Input does not contains values for the properties. Final Output should be a JSON not schema\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=input)
        
        print("------schema---")
        print(_input)

        output = model(_input.to_string())
        print(output)
        output=output.replace("Here is the well-formatted instance of the schema:","").replace("```","").replace("``","")
        print(output)
        
        cptdesc = ToolsDefination.get_descriptions(output,"cptCodes")
        posdesc = ToolsDefination.get_descriptions(output,"posCodes")
        revdesc = ToolsDefination.get_descriptions(output,"revCodes")
        desc_codes= 'NSA Classification Details for the used:'
        if cptdesc!="":
            desc_codes= desc_codes+ "CPT Codes (Current Procedure Terminology):"+cptdesc
        if posdesc!="":
            desc_codes= desc_codes+ "POS Codes (Place of Service):"+posdesc
        if revdesc!="":
            desc_codes= desc_codes+ "Rev Codes (Revenue Codes):"+revdesc


        url = 'http://localhost:7071/GetClassificationClaimDetailsFromAPI'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGci...H86p0xSUChgivAN20E40yox9E1pX1UIE8rdphAjZUKQqiJ52CTz9NmnhfLP-9VhpspdK7pgzT9xQKOYHf-lDbLi8hWe_9LpNnd3X41Rwz8xD94uBERwWFaHPg'
        }
        
        response = requests.post(url, data=output, headers=headers)
        
        final_output=""
        if response.status_code == 200:
            print(response.text)
      
            final_output= response.text
        else:
            final_output="An internal error occured while processing the request"
            print(f"Error: {response.status_code} - {response.text}")
        


        prompt_template = """  
         If isClassified is true, then  Classification reasons will be applicable.
         If isClassified is false, then  Classification reasons will not be applicable.
         Explain in detail about each field of result then finally request is isClassified based on input.

        {input}
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0,model="text-davinci-003"), verbose=True)
        output = desc_codes+ llm_chain.run(final_output)
        return  output
  
    def get_claim_details_from_db(input):
        time.sleep(15)
        dburi = "mssql+pyodbc://unmaskaccount:ykYIGYgUc4kzSBkl4DXy@zenrpasqld01.database.windows.net/Eligibility?driver=ODBC Driver 17 for SQL Server"
        # dburi = "mssql+pyodbc://linked_reader:l1nked@ZCMPZMSQLD101.dev.zelis.com/CMS_DEV_V2?driver=ODBC Driver 17 for SQL Server"
        db = SQLDatabase.from_uri(
            dburi,
            include_tables=['Claims', 'ClaimServices'],      
        )    
        PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=PromptsConstants.SqlTemplate
        )    
        llm = OpenAI(temperature=0.0)
        db_chain = SQLDatabaseChain.from_llm(llm=llm,db=db,prompt=PROMPT, verbose=True)
        output = db_chain.run(input)
        time.sleep(15)
        return output
                

class InputCPTCodes(BaseModel):
    CodeDescription:str = Field(description="Input should be a CPT Code description or CPT Code ")
class InputPOSCodes(BaseModel):
    CodeDescription:str = Field(description="Input should be a POS Code description or POS Code .")
class InputRevCodes(BaseModel):
    CodeDescription:str = Field(description="Input should be a Rev Code description or Rev Code ")
class InputCPTCodesDescription(BaseModel):
      CodeDescription:str = Field(description="should be a description of list of CPT codes or POS codes or RevCodes ")
class ClassificationResponse(BaseModel):
      inputrequest:str=Field(description="Input should be Key Value pair of any parameters ")

tools = [
            Tool.from_function(
                func=ToolsDefination.execute_cpt_csv,
                name="CPTCodes",
                # args_schema=InputCPTCodes,
                description=" Useful to fetch  CPT Codes (Current Procedure Terminology) details like  What is the CPT Code for Eustachian Tuboplasty ? Answer:: CPT Code 69965;Give me or fetch the CPT Code for Eustachian Tuboplasty ? Answer:: CPT Code 69965. This Tool does not know about NSA Classification  ",
            ),
             Tool.from_function(
                func=ToolsDefination.execute_pos_csv,
                name="POSCodes",
                # args_schema=InputPOSCodes,
                description="Useful to fetch  POS Code (Place of Service) details  like What is the POS Code for A facility whose primary purpose is education ? Answer:: Pos Code: 3; Give me or fetch the POS Code for A facility whose primary purpose is education ? Answer::POSCode: 3. Return POSCode if user asks for POSCode not description.",
            ),
            Tool.from_function(
                func=ToolsDefination.execute_rev_csv,
                name="RevCodes",
                # args_schema=InputRevCodes,
                description="Useful to fetch  Rev Code (Revenue Codes) details like What is the Rev Code for All-inclusive room and board ? Answer:: Rev Code 111; Give me or fetch the Rev Code for All-inclusive room and board ? Answer:: Rev Code 111. This Tool does not know about NSA Classification ",
            ),
            Tool.from_function(
                func=ToolsDefination.get_claim_details_from_db,
                name="claim_details_database",
                description="Useful When you need to fetch  Claim details from database in order to make request and call NSA Classification tool. Input should be claimId or PayorClaimNumber. Use other Tools to get CPT Codes, Rev Codes and POS Codes based on their descriptions. ",
            ),
             Tool.from_function(
                func=ToolsDefination.NSAClassification,
                name="NSA_Classification",
                # args_schema=ClassificationResponse,
                return_direct=True,
                description="Useful When you need to get NSA Classification details or to verify is Classified with Classification Reasons. Input parameters should be always seperated by (;) like parameter1:value1,value2; parameter2:value3; parameter3:value4. First Priority of Input is output from other Tools, then second Priority is User Input. Always try to use complete output of other tools",
            ),
        ]
class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    # args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        self.return_direct=True
        return 5

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")
class NSAClassification(BaseTool):
    name = "NSAClassification"
    description = "useful for when you need to verify NSA Classification"
    # args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "is Classified"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")    
# tools = [ CustomCalculatorTool(),NSAClassification() ]    
class ChatbotTools:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
  
    def setup_agent(self):
        # Setup LLM and Agent

        # prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
        #             You are terrible at NSA Classification Reason Errors.
        #             You have access to a multiple tools:"""
        
        # suffix = """
        # You should call Tools based on the Tool description.
        

        
        # Begin!"

        # {chat_history}
        # Question: {input}
        # {agent_scratchpad}"""

        # prompt = ZeroShotAgent.create_prompt(
        #     tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["input", "chat_history", "agent_scratchpad"],
        # )

        # if "memory" not in st.session_state:
        #     st.session_state.memory = ConversationBufferMemory(
        #         memory_key="chat_history"
        #     )

        # llm_chain = LLMChain(
        #     llm=OpenAI(
        #         temperature=0, max_retries=1, max_tokens=500,model_name="gpt-3.5-turbo"
        #     ),
        #     prompt=prompt,
        # )
        # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=tools, verbose=True,handle_parsing_errors=True, memory=st.session_state.memory
        # )
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",max_retries=1, temperature=0.0)
        agent_chain = initialize_agent(
        tools=tools,
        memory =ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        
        
    )
        agent_chain.prep_inputs("Agent should always use exact match codes from CPTCodes,POSCodes and RevCodes Tools. ")
        # agent_chain.prep_inputs("AGent does not know about maths. Always use the CustomCalculatorTool ")
       
#         agent_chain.agent.llm_chain.prompt.template="""Have a conversation with a human, answering the following questions as best yo  can based on the context.

# You have access to a multiple tools and you does not know about CPTCodes,POSCodes and RevCodes.

# CPTCodes: Useful to fetch  CPT Codes (Current Procedure Terminology) details like  What is the CPT Code for Eustachian Tuboplasty ? Answer:: CPT Code 69965; Give me description of CPT Code 69965? Answer:: Eustachian Tuboplasty. 
# POSCodes: Useful to fetch  POS Code (Place of Service) details  like What is the POS Code for A facility whose primary purpose is education ? Answer:: Pos Code 3; Give me description of POS Code 3 ? Answer:: A facility whose primary purpose is education.
# RevCodes: Useful to fetch  Rev Code (Revenue Codes) details like What is the Rev Code for All-inclusive room and board ? Answer:: Rev Code 111; Give me description of Rev Code 111? Answer:: All-inclusive room and board.

# claim_details_database: Useful When you need to fetch all Claim details from database in order to make request and call NSA Classification tool. Input should be claimId or PayorClaimNumber.
        
# NSA_Classification: Useful When you need to get NSA Classification details or to verify is Classified with Classification Reasons. First Priority of Input is output from other Tools with all parameters in key value pair, then second Priority is User Input. Doesn't know about CPT Code descriptions, POS Code Descriptions and Rev Code descriptions.

# Use the following format:

# Question: the input question you must answer 
# Thought: you should always think about what to do
# Action: the action to take, should be one of [CPTCodes, POSCodes, RevCodes, claim_details_database, NSA_Classification]
# Action Input: the input to the action
# Observation: the result of the action

# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

#             Begin!"

           
#             Question: {input}
#             {agent_scratchpad}"""
        return agent_chain
        
     

    @utils.enable_chat_history
    def main(self):
        agent = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                response = agent.run(user_query, callbacks=[st_cb])
                print(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response}) 
                st.write(response)               

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
  
    