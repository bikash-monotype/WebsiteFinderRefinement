from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import os

load_dotenv()

os.environ['OPENAI_API_KEY'] = 'sk-1234567890abcdef1234567890abcdef'

default_llm = ChatOpenAI(
    model='llama3.1:8b',
    base_url='http://localhost:11434/v1/',
)

marketing_analyst = Agent(
    role='Marketing Analyst',
    goal='Analyze the effectiveness of marketing campaigns in driving sales for a specific product line.',
    verbose=True,
    llm=default_llm,
    model_name='llama3.1:8b',
    max_iter=20,
    time_limit=300,
    backstory=(
        """
        As a skilled Marketing Analyst with a strong background in data analysis and campaign performance evaluation, you specialize in deciphering complex marketing data to draw actionable insights. Your expertise enables you to understand nuanced trends and the impacts of various marketing strategies on sales figures.
        You utilize advanced analytical tools and methodologies to ensure precise and insightful analysis. Your work supports strategic decision-making by providing clear evidence of what marketing approaches are working and which are not.
        With your experience, you provide comprehensive reports that help shape future marketing strategies and optimize advertising spends. Your analytical skills are crucial in maximizing ROI for marketing efforts.
        """
    )
)

campaign_effectiveness_task = Task(
    description=(
        """
        Evaluate the data provided below to determine the impact of the recent marketing campaign on the sales of our flagship product, the X120 Drone. 

        **Marketing Data:**
        - Campaign Duration: 3 months
        - Total Spend: $200,000
        - Channels Used: Social Media (50%), Television (20%), Online Ads (30%)
        - Sales Increase: 15% during the campaign period compared to the previous quarter

        Using the information provided, assess whether the investment in the marketing campaign was justified in terms of sales increment. Consider the cost of the campaign, the sales growth, and the channels used. Provide a detailed analysis of which marketing channel was the most effective and suggest any improvements or changes for future campaigns.
        
        Be meticulous in your analysis and provide a clear, data-backed conclusion. Also, consider any external factors that might have influenced the sales beyond the marketing efforts.
        """
    ),
    agent=marketing_analyst,
    expected_output=(
        """
        ['Campaign Justification: Yes/No', 'Most Effective Channel: Channel Name', 'Recommended Improvements: Specific Suggestions']
        """
    )
)

campaign_crew = Crew(
    agents=[marketing_analyst],
    tasks=[campaign_effectiveness_task],
    verbose=True,
    allow_delegation=False
)

results = campaign_crew.kickoff()
print(results)