# Project Backgroud
The Tampa Bay Rays Organization is one of the smallest market teams in the MLB and consistently has one of the lowest known payrolls for its team. They recently had a change in ownership when Stuart Sternberg sold the franchise late last year to Patrick Zalupski. The Rays' market size makes them value their ability to trade for traits and key pieces on the team more than the average MLB team. They made it to the World Series championship in 2020 and followed it with some strong seasons in 2021-2023, but have had a string of 2 consecutive down years since.

## Business Problem & Goal of Analysis

**1) Gap between on-field success and box office success**

The Tampa Bay Rays represent one of the clearest examples of a team that wins consistently but struggles to translate that success into attendance. Over the past decade, the Rays have made multiple playoff appearances and even reached the World Series, yet they regularly rank near the bottom of the MLB in attendance.

Other small-market teams with less success still generate stronger attendance, meaning the Rays’ issue is more structural than performance-based.

**2) Financial impact of empty seats beyond ticket revenue**

Unsold tickets are financially difficult not just for ticket revenue, but for the whole experience. A fan attending a game typically pays for parking, and might buy some food and merchandise as well. Without that seat filled, those other revenue streams dry up as well.

Beyond direct spending, a full stadium creates energy and atmosphere, which enhances the experience for other fans and makes games more attractive to sponsors and broadcasters. Low attendance weakens the environment overall and reduces the odds of return customers.

**3) Geographic and industry factors affecting attendance**

The Rays' current stadium, Tropicana Field, is located in St. Petersburg, while much of the Tampa Bay population and business activity is centered across the bay in Tampa. The logistics of rush hour traffic across the bridges make it very difficult. Other teams have better city transportation infrastructure and the stadium is better located.

**4) Description of the Organization**

The Tampa Bay Rays Organization is one of the smallest market teams in the MLB and consistently has one of the lowest known payrolls for its team. They recently had a change in ownership when Stuart Sternberg sold the franchise late last year to Patrick Zalupski. The Rays' market size makes them value their ability to trade for traits and key pieces on the team more than the average MLB team. They made it to the World Series championship in 2020 and followed it with some strong seasons in 2021-2023, but have had a string of 2 consecutive down years since.

**5) Business Challenge & Impact**

The challenge that the Rays face currently is their poor attendance and ticket sales for their games. As we have mentioned already, they have some issues relating to team success, the stadium switching locations farther from higher population while losing stadium capacity, the market of Tampa in general, and pricing. Being in a rebuilt stadium starting this spring, and eventually the new stadium in 2029 will help this revenue issue a lot, but it seems it cannot come soon enough.

The team seems to be able to create revenue and deliver value, but the ceiling on them in 2024 and 2025 was harmful since they couldn't physically sell more tickets. In Tropicana Stadium now in 2026, they don't yet offer as much opportunity for luxury seating, which usually can lend to the margin and attendance overall as well.


### Data Sources & References
1. Dataset Used: teamstat.csv, gameinfo.csv, Salaries.csv, MLB_payrolls.csv

2. In this analysis we will be using 4 dataset where we will combine the datasets **teamstat.csv** and **gameinfo.csv**. The reason to why we will be combining the two dataset is because **teamstat** include columns like contains team-level statistics - line scores, lineups, and team statistics and **gameinfo.csv** contains game-level information such as teams, attendance, umpires, etc both of these will be useful to our analysis. In addition to that, we will look to combine the datasets Salaries.csv and MLB_payrolls.csv since the Salaries.csv ranges salaries from 1985 to 2016 and MLB_payrolls ranges from 2011-2024.


## Merging the Gameinfo and Teamstats Datasets

We will merge the datasets gameinfo and teamstats where we will use the gameinfo gid (game's id) as the main dataset and the teamstats as a supplement.

**Gameinfo structure**: Gameinfo counts each game once and then it dictates who was at home and who was the visitor.
**Teamstats structure**: Another issue with teamstats is that it is a Team level dataset (501,800 rows). Each game has 2 teams: Home team and Away team.
**Game level focus**: Since the focus of this analysis is on game level stats, we will use a Python script where if a gameid matches from teamstats with gamestats. If gids match we will keep those columns.
**Same data source**: It is important to note that both these data come from the same source, therefore follows the same rules.
**Team perspective recording**: Another major point with the teamstat dataset is that it is recorded from the perspective of the team. For example, b_r and b_hr is the team that is at home.

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/d5d6dde4-d339-4b05-871e-7d95398b93da" />
<img width="868" height="393" alt="image" src="https://github.com/user-attachments/assets/4859c21f-916e-466f-8602-3b6febcbdb1f" />


<img width="1027" height="552" alt="image" src="https://github.com/user-attachments/assets/4b8fedb2-d811-4480-b459-cf95ac2bf27f" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/a914d518-01df-42d6-acc4-7b909ff0e299" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/b70b14ec-2ff3-4d50-a151-79f5b580d6ac" />
<img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/58522de4-314b-4220-bccc-bebf8a311e27" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/96c16073-5d21-4395-863d-3f84e124af5d" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/96ca9821-2919-4a1e-8cf7-137bceb16c6f" />
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/84098229-0b43-4577-b62a-ca762f2419ca" />
<img width="945" height="590" alt="image" src="https://github.com/user-attachments/assets/65d29441-bb4d-48b9-9499-670b6124e393" />
<img width="945" height="590" alt="image" src="https://github.com/user-attachments/assets/dfa5d9e7-9a3c-4a29-b0fe-6e594e617886" />
<img width="1023" height="393" alt="image" src="https://github.com/user-attachments/assets/6a4953e2-1d1f-41b9-b7e0-18eb7c81a364" />


<p>The datasets that covers 2,250 from 1998–2025 but we had to stop at 2020 in the data since they are changing the stadium because of the hurricane and covid, it features weather (temp, precipitation, sky conditions), game timing (day/night, day of week, weekend), team performance (win/loss record, runs scored, home runs), opponent, and season. The target variable is attendance (avg: ~17,221; range: 2,924–55,000)</p>

<h3 style="color: #2c3e50;">1. Model Selection</h3>
<p>The task is to predict attendance for the model selection, in consideration of three points which are: Interpretability vs. predictive power. Also The Rays' front office needs to schedule more weekend games, the dataset is mixed up with feature types, (temperature, win/loss record, run differential) with categorical ones (opponent, day of week, sky condition, game type).</p>

<h3 style="color: #2c3e50;">2. Models</h3>

<p><strong>Model 1 – Multiple Linear Regression (Baseline):</strong> The starting point of model 1 to be tested is Multiple Linear Regression (Baseline), which is clean and easy to estimate attendance such as: "A weekend game is worth +X fans also it shows the season record adds +Y which it shows how amazing and beats every other model well.</p>

<p><strong>Model 2 – Random Forest Regressor:</strong> The prediction of the build of hundreds of decision trees in Random Forest Regressor sets the data to average prediction, it also helps reduce the variance that makes single trees unstable. The tool is telling the Rays which of (opponent, month, weather, team record) drive attendance most, which it can separate missing values without needing extensive imputation.</p>

<p><strong>Model 3 – Gradient Boosting Regressor (XGBoost):</strong> Where Random Forest builds trees in parallel, what XGBoost does is build each tree in order, which it can fix residual errors. This type of order predicts accuracy among the three models, it can also prevent overfitting on a dataset of this size (~2,200 complete records).</p>

<h3 style="color: #2c3e50;">3. Train/Test Split</h3>
<p>Because the nature of the games has been ordered from 1998–2025, a standard random 80/20 split would cause data breach, so the approach is to train a model on 2020 games and tested on 2010 games which it doesn't make any sense. Instead the two strategies will be used so we will split the seasons 1998–2021 which are 85% of the games and for training we will use 2022–2025 as the final test to mirror the real world using historical patterns.</p>


<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/0780e8c1-5d7c-4f39-a067-5ed1c083e6dd" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/cd6d9435-2440-47e0-aa5d-6543636aaea2" />
<img width="1389" height="574" alt="image" src="https://github.com/user-attachments/assets/b10617f1-f341-4dd2-99a6-cc143adc0a03" />
<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/40185bd7-8585-4432-885d-29a3ddf4eadd" />
div style="border: 2px solid #4a90d9; border-radius: 8px; padding: 20px; background-color: #f9f9f9; font-family: Arial, sans-serif; max-width: 900px;">

<h3 style="color: #2c3e50;">1. Top Features by Gain (The "Impact" Players)</h3>
<p>Gain measures how much a feature improves the model's accuracy, lets take this step by step when we use the model features and make a split it reduces errors and your attendance prediction.</p>
<ul>
  <li><strong>gametype_x_regular:</strong> This is the most powerful feature. It suggests that whether a game is a "regular" game vs. an exhibition or special event is the single biggest factor in swinging the attendance numbers.</li>
  <li><strong>The "Big Names" (opp_NYA, opp_BOS):</strong> In sports modeling, certain teams (like the Yankees or Red Sox) are massive draws because the model sees that its a high profile team it will immediately predict that the stadium is packed.</li>
  <li><strong>is_weekend &amp; day_of_week_Saturday/Sunday:</strong> These are high-impact features. The model finds that the jump in attendance from a Tuesday to a Saturday is very predictable and significant.</li>
</ul>

<h3 style="color: #2c3e50;">2. Top Features by Weight (The "Workhorses")</h3>
<p>Weight (or Frequency) measures how many times a feature is used to split the data across all the trees in your model. This represents "quantity."</p>
<ul>
  <li><strong>season and month:</strong> the model always uses it as anchor points before even looking at who is playing because when it knows when they are playing it can predict it for a trend such as summer months because generally they have higher attendance than early spring.</li>
  <li><strong>parsed_starttime:</strong> This is a high-frequency splitter. The model is constantly checking the time of day to fine-tune the prediction, even if the "Gain" (the specific impact of one hour vs. another) isn't as massive as a weekend vs. weekday.</li>
</ul>

<h3 style="color: #2c3e50;">3. The "Why the Difference?" Insight</h3>
<p>You'll notice that season is #1 in Weight but much lower in Gain. Conversely, gametype_x_regular is #1 in Gain but doesn't even appear in the top 15 for Weight. Here is what that means for your data:</p>
<ul>
  <li><strong>Broad Filters (High Weight):</strong> The model uses season and month to sort the data over and over again. They are necessary for the model's structure, but they don't always provide the "final" answer.</li>
  <li><strong>Decision Makers (High Gain):</strong> Features like is_weekend or opp_NYA are the "final word." Once the model knows it's a Saturday game against a popular opponent, it can make a very high-confidence prediction. It doesn't need to split on these features many times to get the point.</li>
</ul>

<h3 style="color: #2c3e50;">Summary for your Attendance Model</h3>
<p>If you want to increase attendance, your model suggests you should focus on:</p>
<ol>
  <li><strong>Scheduling/Marketing:</strong> Capitalize on the weekend/Saturday surge.</li>
  <li><strong>Opponent Strategy:</strong> You know exactly which opponents (NYA, BOS, ATL) drive the most "predictable" spikes in revenue.</li>
  <li><strong>Seasonality:</strong> Use the season and month trends as your baseline, but look to the "Gain" features to see what actually moves the needle beyond just the time of year.</li>
</ol>

</div>
<img width="987" height="390" alt="image" src="https://github.com/user-attachments/assets/3fd8119c-16e5-4a14-9aab-656310ba296e" />

h2 style="color: #2c3e50;">Business Recommendations for Improving Tampa Bay Rays Home Attendance</h2>

<p>To boost Home attendance in the Tampa Bay Rays we went by our comprehensive analysis of game-level data, that was optimized from the XGBoost model and identified several key areas.</p>

<h3 style="color: #2c3e50;">1. Strategic Scheduling &amp; Game Timing</h3>
<ul>
  <li><strong>Prioritize Weekend &amp; Evening Games:</strong> from our analysis we have collected data about when is the higher attendance which are on Saturday and Sunday games also the exact times which are 5 PM and 8 pm that mostly have more high-profile games.</li>
  <li><strong>Leverage Early &amp; Late Season:</strong> the excitement of the start and the end of the season which makes the fans attend more games which has been documented in March and October.</li>
</ul>

<h3 style="color: #2c3e50;">2. Opponent-Based Marketing &amp; Promotions</h3>
<ul>
  <li><strong>Target Popular Opponents:</strong> Teams like the New York Yankees (NYA), Boston Red Sox (BOS), Atlanta Braves (ATL), Chicago Cubs (CHN), and San Francisco Giants (SFN) consistently draw larger crowds which means more marketing and premium pricing especially for games against these popular opponents.</li>
  <li><strong>Bundle Less Popular Games:</strong> creating promotion for families also ticket packages for games that have lower attendance mainly in middle of the week games.</li>
</ul>

<h3 style="color: #2c3e50;">3. Enhance Game Day Experience &amp; Value</h3>
<ul>
  <li><strong>Focus on 'Event' Games:</strong> Playoff games like the World Series, LCS actually draws the biggest crowds. Most fans come to these games because its the most important ones, from these events the team should take ideas and apply them on regular season games to make the other games feel the energy of the most important games by adding giveaways and promotions like season pass.</li>
  <li><strong>Monitor Game Performance, But Don't Overreact to Single Games:</strong> Our analysis suggests that the loss in the previous games does not significantly deter attendance for the next home game. This shows that the fan base's decision is influenced by broader factors rather than immediate team performance fluctuations.</li>
</ul>

<h3 style="color: #2c3e50;">4. Continuous Analysis &amp; Adaptation</h3>
<ul>
  <li><strong>Dynamic Pricing &amp; Promotion:</strong> adjusting pricing accordingly from monitoring attendance trends it gives leverage insight to the model forecast expecting attendance.</li>
  <li><strong>Explore Weather-Related Strategies:</strong> Weather factors might affect attendance such as heavy rain and discourage travel because the Tropicana Field is a dome, we need to investigate more to give the fans comfort and accessibility to the stadium.</li>
</ul>

<h3 style="color: #2c3e50;">5. Payroll Investment vs. Attendance (Long-Term Consideration)</h3>
<ul>
  <li>Payroll investment and attendance is not connected directly to each other simply teams always linked to pay roll while attendance is not if you pay more doesn't mean you will have higher attendance, marketing has to be involved so both categories can link together for a better strategic investment.</li>
</ul>

</div>
















