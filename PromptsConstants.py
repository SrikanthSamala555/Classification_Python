
SqlTemplate="""Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.,
    Use the following format:
    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery in the form of JSON"
    Answer: "Final answer here"   
    Use only below Tables and columns for queries   
    
    CREATE TABLE [dbo].[Claims](
    [ClaimID] [int]  NOT NULL Primary Key,
	[PayorID] [int] NOT NULL,
    [ClaimTypeID] [int] NOT NULL,
    [FromDateOfService] [smalldatetime] NULL,
    [PayorClaimNumber] [varchar](50) NULL 
    )
    CREATE TABLE [dbo].[ClaimTypes]
    (
    [ClaimTypeID] int IDENTITY(1,1) NOT NULL Primary Key,
    [ClaimType] VARCHAR(50)
    )

CREATE TABLE [dbo].[ClaimServices](
    [ClaimServiceID] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL Primary Key,
    [ClaimID] [int] NOT NULL,   
    [PlaceOfService] [char](2) NULL,    
    [ProcedureCode] [varchar](10) NULL,
    [RevenueCode] [char](4) NULL,
    [Modifier1] [char](2) NULL
	)  
ALTER TABLE [dbo].[ClaimServices]  WITH CHECK ADD  CONSTRAINT [FK_ClaimServices_Claims] FOREIGN KEY([ClaimID])
REFERENCES [dbo].[Claims] ([ClaimID])
ON DELETE CASCADE
ALTER TABLE [dbo].[Claims]  WITH CHECK ADD  CONSTRAINT [FK_Claims_ClaimTypes] FOREIGN KEY([ClaimTypeID])
REFERENCES [dbo].[ClaimTypes] ([ClaimTypeID])
GO
    Generate the result with the columns:  ClaimID,PayorID,ClaimTypeID,ClaimType,FromDateOfService as EffectiveDate,PayorClaimNumber,PlaceOfService,ProcedureCode,RevenueCode,Modifier1 from the tables.  
    If same column name exists in multiple table, then priority is to use from Claims Table.
    Use '' instead of 'None' , Use yyyy-mm-dd format for dates
    For the same claimId  don't generate multiple rows,  ProcedureCode,  RevenueCode and PlaceofService should be (,) seperated for that column like claimId 1334 with two rows, distinct posCodes ('333','4444'), distinct revCodes ('44','33') and distinct posCodes ('3','44').
    Question: {input} """

cpt_template = """
        You are a helpful AI assistant named Robby.  The user gives you  its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context
        
        context: {context}
        =========
        question: {question}
        ======
        answer:
        """

pos_template = """
        You are a helpful AI assistant named Robby.  The user gives you  its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context
        
        context: {context}
        =========
        question: {question}
        ======
        answer:
        """

rev_template = """
        You are a helpful AI assistant named Robby. The user gives you  its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context
        
        context: {context}
        =========
        question: {question}
        ======
        answer:
        """