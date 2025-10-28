# --- 1. IMPORTAÇÕES DAS BIBLIOTECAS ---
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Importações principais do LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool # Importação corrigida

# Importações para a Ferramenta PANDAS (Agente de Dados)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Importações para a Ferramenta RAG (Agente de Conhecimento)
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# --- 2. CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="Agente de Análise Fiscal (NF-e)",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Agente de Análise Fiscal (NF-e)")
st.caption("IA especialista em análise de Notas Fiscais da União")

# Define o LLM (Modelo de Linguagem) que nossos agentes usarão
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-3.5-turbo-1106",
    api_key=st.secrets["OPENAI_API_KEY"]
)

# --- 3. FUNÇÕES DE CARREGAMENTO E CACHE DE DADOS ---

@st.cache_data(show_spinner="Carregando e processando dados...")
def load_data(periodo: str) -> pd.DataFrame:
    """
    Carrega, funde e retorna um DataFrame consolidado para o período selecionado.
    Os arquivos Excel são lidos da pasta /data.
    """
    st.write(f"Iniciando carregamento para o período: {periodo}...")
    
    file_map = {
        "Junho": ["202506_NFe_NotaFiscal.xlsx", "202506_NFe_NotaFiscalItem.xlsx"],
        "Julho": ["202507_NFe_NotaFiscal.xlsx", "202507_NFe_NotaFiscalItem.xlsx"],
        "Agosto": ["202508_NFe_NotaFiscal.xlsx", "202508_NFe_NotaFiscalItem.xlsx"],
    }
    
    data_path = Path("data")
    all_dfs = [] 

    meses_para_carregar = []
    if periodo == "Consolidado (Todos os Meses)":
        meses_para_carregar = ["Junho", "Julho", "Agosto"]
    elif periodo in file_map:
        meses_para_carregar = [periodo]
    else:
        return pd.DataFrame()

    for mes in meses_para_carregar:
        try:
            file_nota = data_path / file_map[mes][0]
            file_item = data_path / file_map[mes][1]
            
            if not file_nota.exists() or not file_item.exists():
                st.error(f"Arquivos para o mês de {mes} não encontrados na pasta 'data/'.")
                continue

            df_notas = pd.read_excel(file_nota, engine='openpyxl')
            df_itens = pd.read_excel(file_item, engine='openpyxl')
            
            st.write(f"[{mes}] Arquivo de Notas carregado: {df_notas.shape[0]} linhas.")
            st.write(f"[{mes}] Arquivo de Itens carregado: {df_itens.shape[0]} linhas.")

            colunas_para_remover_itens = [
                'MODELO', 'SÉRIE', 'NÚMERO', 'NATUREZA DA OPERAÇÃO', 'DATA EMISSÃO',
                'CPF/CNPJ Emitente', 'RAZÃO SOCIAL EMITENTE', 'INSCRIÇÃO ESTADUAL EMITENTE',
                'UF EMITENTE', 'MUNICÍPIO EMITENTE', 'CÓDIGO ÓRGÃO SUPERIOR DESTINATÁRIO',
                'ÓRGÃO SUPERIOR DESTINATÁRIO', 'CÓDIGO ÓRGÃO DESTINATÁRIO', 'ÓRGÃO DESTINATÁRIO',
                'CNPJ DESTINATÁRIO', 'NOME DESTINATÁRIO', 'UF DESTINATÁRIO',
                'INDICADOR IE DESTINATÁRIO', 'DESTINO DA OPERAÇÃO', 'CONSUMIDOR FINAL',
                'PRESENÇA DO COMPRADOR'
            ]
            
            colunas_existentes_para_remover = [col for col in colunas_para_remover_itens if col in df_itens.columns]
            df_itens_limpo = df_itens.drop(columns=colunas_existentes_para_remover)
            
            df_consolidado_mes = pd.merge(
                df_notas,
                df_itens_limpo,
                on="CHAVE DE ACESSO",
                how="left"
            )
            
            st.write(f"[{mes}] Merge concluído. DataFrame consolidado: {df_consolidado_mes.shape[0]} linhas.")
            all_dfs.append(df_consolidado_mes)

        except Exception as e:
            st.error(f"Erro ao processar os arquivos de {mes}: {e}")
            return pd.DataFrame() 

    if not all_dfs:
        st.warning("Nenhum dado foi carregado.")
        return pd.DataFrame()
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    st.success(f"Sucesso! DataFrame final consolidado com {final_df.shape[0]} linhas e {final_df.shape[1]} colunas.")
    return final_df


# --- 4. FUNÇÕES DE CRIAÇÃO DO AGENTE E FERRAMENTAS ---

@st.cache_resource(show_spinner="Configurando Agente de Análise de Dados...")
def create_pandas_agent_tool(_df: pd.DataFrame) -> AgentExecutor:
    """Cria a Ferramenta 1: O Agente de Análise de Dados (Pandas)."""
    
    PREFIXO_PROMPT = """
    Você é um agente especialista em análise de dados com Pandas.
    Sua resposta DEVE ser em Português do Brasil.
    Ao gerar gráficos usando plotly, salve-os como arquivos JSON e informe o caminho.
    """
    
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        _df,
        prefix=PREFIXO_PROMPT,
        verbose=True, 
        agent_type=AgentType.OPENAI_FUNCTIONS, 
        allow_dangerous_code=True
        # O PARÂMETRO 'handle_parsing_errors=True' FOI REMOVIDO DAQUI
    )
    return pandas_agent

@st.cache_resource(show_spinner="Configurando Agente Consultor Fiscal...")
def create_rag_retriever_tool() -> any: # Retorna um 'BaseTool'
    """Cria a Ferramenta 2: O Agente de Conhecimento (RAG)."""
    
    # 1. Carregar nosso arquivo de conhecimento
    loader = TextLoader("kb/fiscal_knowledge.txt", encoding="utf-8")
    documents = loader.load()
    
    # 2. Dividir o texto em "pedaços" (chunks)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 3. Criar "Embeddings" (vetores) para cada pedaço
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    
    # 4. Criar o Banco de Dados Vetorial (FAISS) e o Retriever
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Erro ao criar o vector store: {e}")
        return None

    # 5. Criar a Ferramenta de Retriever
    rag_tool = create_retriever_tool(
        retriever,
        "consultor_fiscal", # Nome da ferramenta
        "Consulta a base de conhecimento sobre legislação fiscal, CFOP, NCM e regras de negócio. Use para perguntas conceituais."
    )
    return rag_tool

# (Função 'create_master_agent' da nossa correção anterior - sem alterações)
def create_master_agent(pandas_agent_executor: AgentExecutor, rag_tool: any) -> AgentExecutor:
    """
    Cria o Agente Orquestrador (Principal) que decide qual ferramenta usar.
    """
    
    pandas_tool = Tool(
        name="analista_de_dados_pandas",
        description=(
            "Use esta ferramenta para responder perguntas quantitativas sobre os dados das notas fiscais (DataFrame). "
            "Exemplos: 'Qual o fornecedor com mais gastos?', 'Qual o valor total das notas de Junho?', "
            "'Gere um gráfico de pizza dos 5 principais NCMs', 'Qual a média do VALOR TOTAL?'"
        ),
        func=pandas_agent_executor.invoke 
    )
    
    tools = [pandas_tool, rag_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente especialista em análise fiscal. Você tem duas ferramentas: \n"
                "1. `analista_de_dados_pandas`: Usado para analisar os dados numéricos (DataFrame) das notas fiscais. \n"
                "2. `consultor_fiscal`: Usado para responder perguntas conceituais sobre leis fiscais (CFOP, NCM, etc.). \n\n"
                "Decida qual ferramenta é a mais apropriada. Se a pergunta for mista (ex: 'Este CFOP 5102 está correto para o item X?'), "
                "você pode usar as duas ferramentas em sequência. Responda sempre em Português do Brasil."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor


# --- 5. LÓGICA PRINCIPAL DA INTERFACE (STREAMLIT UI) ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "master_agent" not in st.session_state:
    st.session_state.master_agent = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

with st.sidebar:
    st.header("Configuração da Análise")
    periodo_selecionado = st.selectbox(
        "Selecione o período de análise:",
        ("Selecione um período", "Junho", "Julho", "Agosto", "Consolidado (Todos os Meses)")
    )

if periodo_selecionado != "Selecione um período":
    st.session_state.df = load_data(periodo_selecionado)
    
    if not st.session_state.df.empty:
        pandas_agent_tool = create_pandas_agent_tool(st.session_state.df)
        rag_tool = create_rag_retriever_tool()
        
        if rag_tool:
            st.session_state.master_agent = create_master_agent(
                pandas_agent_executor=pandas_agent_tool, 
                rag_tool=rag_tool
            )
        else:
            st.error("Não foi possível inicializar o Consultor Fiscal (RAG). Verifique o arquivo kb/fiscal_knowledge.txt.")
            
else:
    st.session_state.master_agent = None
    st.session_state.chat_history = []
    st.session_state.df = pd.DataFrame()
    st.info("Por favor, selecione um período na barra lateral para iniciar a análise.")


# --- Interface de Chat ---
if st.session_state.master_agent:
    # Exibe as mensagens do histórico
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura a pergunta do usuário
    if user_question := st.chat_input("Faça sua pergunta ao agente..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    agent_input = {
                        "input": user_question,
                        "chat_history": [
                            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                            for msg in st.session_state.chat_history[:-1]
                        ]
                    }
                    
                    response = st.session_state.master_agent.invoke(agent_input)
                    resposta_final = response['output']
                    st.markdown(resposta_final)
                    st.session_state.chat_history.append({"role": "assistant", "content": resposta_final})
                
                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
else:
    # Mensagem se nenhum agente estiver carregado
    # A LINHA DA IMAGEM QUEBRADA FOI REMOVIDA DAQUI
    st.markdown("### Selecione um período na barra lateral à esquerda para carregar o agente e iniciar sua análise.")