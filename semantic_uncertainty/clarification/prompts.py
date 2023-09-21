GENERIC_QUESTION_PROMPT = 'This is a conversation between a user and a question-answering bot. The bot only returns the shortest correct answer to the question itself without forming a full sentence.\nUser: What is the capitcal city of France?\nBot: Paris.\nUser: On what date did Neil Armstrong land on the moon?\nBot: July 20, 1969. \nUser: {initial_question}\nBot:'


GENERIC_QUESTION_PROMPT_CLAQUA = '''This is a conversation between a user and a question-answering bot. The bot only returns the shortest correct answer to the question itself without forming a full sentence.
User: entity1: Schadonia Schadonia is a genus of lichenized fungi in the Ramalinaceae family.
entity2: "Lecanorineae  The Lecanorales are an order of mostly lichen-forming fungi belonging to the class Lecanoromycetes in the division Ascomycota. The order contains 26 families, 269 genera, and 5695 species.
What is higher classification for schadonia <EOS> Lecanorineae <EOS> Which cateloge it should be classified into
Final answer by bot: 190901

User: entity1: Beat'n Down Yo Block Beat'n Down Yo Block! is the debut album from Atlanta based rapper Unk. It was released October 3, 2006. 
entity2: "2econd Season 2econd Season is the second album by Atlanta-based rapper Unk. It was released on November 4, 2008. The album features guest appearances by Sean Kingston, Ray J, Three 6 Mafia and more, while production as handled mainly by Oomp Camp's own DJ Montay.
What is the album after beat'n down yo block? <EOS> 2econd Season <EOS> What was the name of the first release of the album?
Final answer by bot: Beat'n Down Yo Block!

 User: {initial_question}
 
 Final answer by bot:'''

UNIFIED_CLARIFYING_QUESTION_PROMPT = '''This is a conversation between a user and a question-answering bot.
User: On what date did he land on the moon?
Bot: To answer this question, I need to ask the following clarfiying question: Who is he?
User: Which country on this continent has the largest population?
Bot: To answer this question, I need to ask the following clarfiying question: Which continent? 
User: Directors of Two Trains Running
Bot: To answer this question, I need to ask the following clarfiying question: Which one do you mean, the 1991 pre-Broadway theater production of Two Trains Running or  the 2006-2007 theater production Two Trains Running?
User: Harding is named after whom?
User: On what date did he land on the moon?
Bot: To answer this question, I need to ask the following clarfiying question: Who is he?
User: Which country on this continent has the largest population?
Bot: To answer this question, I need to ask the following clarfiying question: Which continent? 
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to Harding County or the lunar crater Harding?

User: {initial_question}\nBot: To answer this question, I need to ask the following clarfiying question:
'''

CLARIFYING_QUESTION_PROMPT = '''This is a conversation between a user and a question-answering bot.
User: On what date did he land on the moon?
Bot: To answer this question, I need to ask the following clarfiying question: Who is he?

###

User: Which country on this continent has the largest population?
Bot: To answer this question, I need to ask the following clarfiying question: Which continent? 

###

User: {initial_question}\nBot: To answer this question, I need to ask the following clarfiying question:'''


UNIFIED_P_TRUE_PROMPT = '''This bot determines whether a given question is ambiguous or not.
User: Who was the first woman to make a solo flight across this ocean?
This question is ambiguous: True.

User: Who was the first woman to make a solo flight across the Atlantic?
This question is ambiguous: False. 

User: In which city were Rotary Clubs set up in 1905?
This question is ambiguous: False.

User: Who along with Philips developed the CD in the late 70s?
This question is ambiguous: False. 

User: Where is the multinational corporation based?
This question is ambiguous: True.

User: Is there a sequel to bazil broketail
Bot: A Sword for a Dragon
User: How about the style of this creative work?
This question is ambiguous: True.

User: What is higher classification for schadonia
Bot: Lecanorineae
User: Which cataloge it should be classified into?
This question is ambiguous: True.

User: What is family for 807 ceraskia
Bot: Eos family
User: What is the star system of the object
This question is ambiguous: False.

User: What is division for soo greyhounds
Bot: West division
User: Which is the represented city?
This question is ambiguous: False.

User: Too which city does black mountain belong?
Bot: Henderson. 
User: To which mountain range?
This question is ambiguous: False. 

User: What opera was conducted by elgar howarth? 
Bot: The prince of homburg
User: What is the composition created by the composer?
This question is ambiguous: False.

User: School district for scripps middle school
Bot: Lake Orion Community Schools
User: What is the lowest grade taught in the school?
This question is ambiguous: False.

User: State the name of the person who influenced richard j. bernstein 
Bot: Hannah Grace Strickland
User: Name of the book?
This question is ambiguous: True.

User: Name the album next, after venereology
Bot: Noisembryo: Psycho-Analytic Study of Coital Noise Posture
User: What was its primary release?
This question is ambiguous: True.

User: In wjbk, what is the content that is featured?
Bot: Television
User: Mention the time zone
This question is ambiguous: False.

User: Directors of Two Trains Running
This question is ambiguous: True. 

User: What is the genre of the book About Love?
This question is ambiguous: False.

User: who is The Inferno's developer?
This question is ambiguous: False.

User: Who is executive producer for The Killing Kind?
This question is ambiguous: True.

User: {question}
This question is ambiguous: True'''

# CLARIFYING_QUESTION_PROMPT = 'This is a conversation between a user and a question-answering bot.\nUser: {initial_question}\nBot: To answer this question, I need to ask the following clarfiying question:'

CLARIFYING_INFORMATION_PROMPT = 'This is a conversation between a user and a question-answering bot. The user wants to get an answer to the following question: "{precise_question}"\nUser: {initial_question}\nBot: To answer this question, I need to ask the following clarifying question: {clarifying_question}\nUser:'
CLAQUA_SINGLETURN_CLARIFYING_INFORMATION_PROMPT = '''This is a conversation between a user and a question-answering bot. 

The user wants to get an answer to the following question: "What is the name of the final edited version of The Cheat?  ( After embezzling from a charity, a greedy socialite takes a loan from a Japanese merchant, who has a different payment in mind than just money.)"
User: entity1: The Cheat <S> award.nominated_work broadcast.content film.film media_common.cataloged_instance media_common.creative_work ratings.rated_entity <S> In the docks of Bordeaux a well-known homosexual was murdered. Police Inspector Michel Verta starts investigating, when he falls in love with Bernard, a handsome young musician. This not ...
entity2: The Cheat <S> broadcast.content ratings.rated_entity media_common.creative_work media_common.cataloged_instance film.film award.ranked_item <S> After embezzling from a charity, a greedy socialite takes a loan from a Japanese merchant, who has a different payment in mind than just money.
What is the name of the final edited version of The Cheat? 
Bot: To answer this question, I need to ask the following clarifying question: 
Which The Cheat are you referring to?
User: The cheat which is about a a greedy socialite that embezzels money from a charity and then takes a loan from a Japanese merchant.

The user wants to get an answer to the following question: "Which company is game developer for Rescue  ( Rescue is an arcade game in which the player navigates a helicopter over the open seas to rescue stranded paratroopers from enemy forces and sharks. It was designed by Chris Oberth and published by Stern in 1982.)"
User: entity1: Rescue <S> commerce.consumer_product commerce.product computer.software cvg.computer_videogame games.game media_common.creative_work ratings.rated_entity <S> Rescue is an arcade game in which the player navigates a helicopter over the open seas to rescue stranded paratroopers from enemy forces and sharks. It was designed by Chris Oberth and published by Stern in 1982.
entity2: Rescue <S> cvg.computer_videogame ratings.rated_entity games.game computer.software commerce.product <S> Rescue. It was published by Mastertronic for ZX.
Which company is game developer for Rescue 
Bot: To answer this question, I need to ask the following clarifying question: 
Which Rescue are you referring to?
User: Rescue the arcade game.

The user wants to get an answer to the following question: "{precise_question}"\nUser: {initial_question}\nBot: To answer this question, I need to ask the following clarifying question: {clarifying_question}\nUser:'''

CLAQUA_MULTITURN_CLARIFYING_INFORMATION_PROMPT = '''This is a conversation between a user and a question-answering bot. The user wants to ask the question: {precise_question} 
\nUser: {initial_question}\nBot: To answer this question, I need to ask the following clarifying question: {clarifying_question}\nUser:'
'''

FINAL_ANSWER_PROMPT = '''This is a conversation between a user and a question-answering bot.
The bot only returns the shortest correct answer to the question itself without forming a full sentence.

User: Who was the first woman to make a solo flight across this ocean?
Bot: Which oceran are you referring to?
User: The atlantic
Final answer by bot: Amelia Earhart 

###

User: Where did she live?
Bot:  Who is she?
User: Judi Bench
Final answer by bot: York.

###

User: {initial_question}
Bot: {clarifying_question}
User: {clarifying_information}
Final answer by bot:'''

P_TRUE_PROMPT = '''This bot determines whether a given question is ambiguous or not.

Q: Who was the first woman to make a solo flight across this ocean?
This question is ambiguous: True.

Q: Who was the first woman to make a solo flight across the Atlantic?
This question is ambiguous: False. 

Q: In which city were Rotary Clubs set up in 1905?
This question is ambiguous: False.

Q: Who along with Philips developed the CD in the late 70s?
This question is ambiguous: False. 

Q: Where is the multinational corporation based?
This question is ambiguous: True.

Q: {question}
This question is ambiguous: True'''

P_TRUE_PROMPT_MULITURN_CLAQUA = '''This bot determines whether a given question is ambiguous or not.

Bazil Broketail. Bazil Broketail (1992) is a fantasy novel written by Christopher Rowley. 
A Sword for a Dragon. A Sword for a Dragon is a fantasy novel written by Christopher Rowley.
"How about the style of this creative work?" could refer both to Bazil Broketail and A Sword for a Dragon: True

###

Draco. Draco Lucius Malfoy is a character in J. K. Rowling's Harry Potter series.
Harry potter and the deathly hallows: part ii. The final adventure in the Harry Potter film series follows Harry (Daniel Radcliffe), Ron (Rupert Grint), and Hermione (Emma Watson) as they prepare for a final battle with Lord Voldemort (Ralph Fiennes), who is determined to destroy Harry once and for all.
"Name the super power?" could refer both to Draco and Harry potter and the deathly hallows: part ii: False

###

The Last Straw. The highly anticipated third book in the critically acclaimed and bestselling series takes the art of being wimpy to a whole new level. 
Hard Luck (Diary of a Wimpy Kid book 8). Diary of a Wimpy Kid: Dog Days is a novel written by American author and cartoonist Jeff Kinney, and is the fourth book in the Diary of a Wimpy Kid series. It was released on October 12, 2009 in the USA and October 13, 2009, in Canada. The film, Diary of a Wimpy Kid: Dog Days, released on August 3, 2012, was based on the book and its predecessor, The Last Straw.
"What is the name of the next?" could refer both to The Last Straw and Hard Luck (Diary of a Wimpy Kid book 8): True.

###
Delta Kappa Epsilon. Delta Kappa Epsilon is one of the oldest North American fraternities with 54 active chapters in the United States and Canada.
Chester newell righter. 
"What was the color of the fraternity?" could refer both to Delta Kappa Epsilon and Chester newell righter: False.

###

807 Ceraskia.  807 Ceraskia is a minor planet orbiting the Sun.
Eos family.  The Eos family (adj. Eoan; FIN: 606) is a very large asteroid family located in the outer region of the asteroid belt.
"What is the star system of the object" coul refer both to 807 Ceraskia and Eos family: False.

###

Seeing Things.  Seeing Things is singer-songwriter Jakob Dylan's first solo studio album.
Women + Country.  Women + Country is singer-songwriter Jakob Dylan's second solo studio album.
"Name the track?" could refer both to Seeing Things and Women + Country: True.

###

Franz Krumm Franz Krumm (* 16 October 1909; \u2020 9 March 1943) was a German footballer. 
World war ii. World War II (often abbreviated to WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945. 
"What is the position on the team?" could refer both to Franz Krumm and World war ii: False

###

{question} True'''


P_TRUE_PROMPT_MULITURN_CLAQUA_OLD = '''This bot determines whether a given question is ambiguous or not.
User: Is there a sequel to bazil broketail
Bot: A Sword for a Dragon
User: How about the style of this creative work?
This question could refer to either of the two previous statements: True.

###

User: What is higher classification for schadonia
Bot: Lecanorineae
User: Which cataloge it should be classified into?
This question could refer to either of the two previous statements: True.

###

User: What is family for 807 ceraskia
Bot: Eos family
User: What is the star system of the object
This question could refer to either of the two previous statements: False.

###

User: What is division for soo greyhounds
Bot: West division
User: Which is the represented city?
This question could refer to either of the two previous statements: False.

###

User: Too which city does black mountain belong?
Bot: Henderson. 
User: To which mountain range?
This question could refer to either of the two previous statements: False. 

###

User: What opera was conducted by elgar howarth? 
Bot: The prince of homburg
User: What is the composition created by the composer?
This question could refer to either of the two previous statements: False.

###

User: School district for scripps middle school
Bot: Lake Orion Community Schools
User: What is the lowest grade taught in the school?

This question could refer to either of the two previous statements: False.

###

User: State the name of the person who influenced richard j. bernstein 
Bot: Hannah Grace Strickland
User: Name of the book?
This question could refer to either of the two previous statements: True.

###

User: Name the album next, after venereology
Bot: Noisembryo: Psycho-Analytic Study of Coital Noise Posture
User: What was its primary release?
This question could refer to either of the two previous statements: True.

###

User: In wjbk, what is the content that is featured?
Bot: Television
User: Mention the time zone
This question could refer to either of the two previous statements: False.

### 

User: {question}
This question could refer to either of the two previous statements: True'''

P_TRUE_PROMPT_CLARIQ = '''This bot determines whether a given question is ambiguous or not.
Question: Tell me about Obama family tree.	
This question is ambiguous: False.

Question: TV on computer.
This question is ambiguous: True.

Question: What is Fickle Creek Farm
This question is ambiguous: False.

Question: Tell me about source of the nile.
This question is ambiguous: True.

Question: How to prepare for the GMAT?
This question is ambiguous: False.

Question: Find condos in Florida.
This question is ambiguous: True.

Question: Tell me about american military university.
This question is ambiguous: False.

Question: {question}
This question is ambiguous: True'''


P_TRUE_ZERO_SHOT_PROMPT_CLARIQ = '''
Question: {question}
This question is ambiguous: True'''

CLAQUA_SINGLE_TURN_CLARIFYING_QUESTION_PROMPT = '''This is a conversation between a user and a question-answering bot.

Question: Casting director for Fakers
When you say Fakers, are you referring to the TV movie or the movie?

###

Question: What is name of place where Ernest Pollard was born?
When you say Ernest Pollard, are you referring to the Nebraska Republican politician or the professor of physics and biophysics?

###

Question: Cunningham Elementary\'s rank
Which Cunningham Elementary are you referring to?

###

Question: Who is the host of Room 101?
Which Room 101?

###

Question: Who is executive producer of Primos
Which Primos are you referring to?

###

Question: What is stadium name for Georgia?
Which Georgia are you referring to?

###

Question: The affected area for Hurricane Ophelia
Which Hurricane Ophelia are you reffering to?

###

Question: What is the religion of Saint?
Which Saint are you referring to?

###

Question: What is name of author of Malheur? 
Which Malheur are you referring to?

###

Question: {initial_question}
'''

P_TRUE_PROMPT_SINGLE_TURN_CLAQUA = ''''This bot determines whether a given question is ambiguous or not.

entity1: Rescue <S> commerce.consumer_product commerce.product computer.software cvg.computer_videogame games.game media_common.creative_work ratings.rated_entity <S> Rescue is an arcade game in which the player navigates a helicopter over the open seas to rescue stranded paratroopers from enemy forces and sharks. It was designed by Chris Oberth and published by Stern in 1982.\nentity2: Rescue <S> cvg.computer_videogame ratings.rated_entity games.game computer.software commerce.product <S> Rescue. It was published by Mastertronic for ZX.\n
"Which company is game developer for Rescue" could refer to both entities "Rescue": True.

entity1: Stanford Cardinal <S> american_football.team award.competitor event.agent media_common.cataloged_instance organization.organization sports.school_sports_team sports.team <S> The Stanford Cardinal football program represents Stanford University in college football at the NCAA Division I FBS level and is a member of the Pac-12 Conference's North Division. Stanford, the top-ranked academic institution with a FBS program, has a highly successful football tradition. The team is currently known as the Cardinal, adopted prior to the 1982 season. Stanford was known as the Indians from 1930 to January 1972, and the Cardinals from 1972 through 1981. A student vote in December 1975 to change the nickname to Robber Barons was not approved by administrators.\nentity2: Stanford Cardinal <S> award.competitor award.nominee basketball.team event.agent media_common.cataloged_instance organization.organization sports.school_sports_team sports.team <S> The Stanford Cardinal men's basketball team represents Stanford University in Stanford, California, United States. The school's team currently competes in the Pac-12 Conference. They are coached by Jerod Haase and play their home games at Maples Pavilion.
"Which is official color for Stanford Cardinal?" could refer to both entities "Stanford Cardinal": True.

entity1: Lifestyle <S> book.subject book.news_topic book.magazine_genre book.literary_genre internet.website_category broadcast.genre film.genre tv.genre media_common.subject media_common.media_genre media_common.catalog_category <S> Lifestyle is the interests, opinions, behaviours, and behavioural orientations of an individual, group, or culture. The term was introduced by Austrian psychologist Alfred Adler with the meaning of a person's basic character as established early in childhood, for example in his 1929 book The Case of Miss R.. The broader sense of lifestyle as a way or style of living has been documented since 1961. Lifestyle is a combination of determining intangible or tangible factors. Tangible factors relate specifically to demographic variables, i.e. an individual's demographic profile, whereas intangible factors concern the psychological aspects of an individual such as personal values, preferences, and outlooks.\nentity2: Lifestyle <S> commerce.consumer_product commerce.product media_common.cataloged_instance media_common.creative_work music.album ratings.rated_entity <S> Lifestyle is the seventh studio album by American indie rock band Silkworm. It was released on August 8, 2000 by the independent record label Touch and Go Records, making it their second on that label. After 1998's self-produced Blueblood, the band's friend and longtime recording engineer Steve Albini again returned to record the album, which was also produced by his girlfriend Heather Whinna.
"which magazine belongs to Lifestyle?" could refer to both entities "Lifestyle": False

entity1: Aggrenox <S> commerce.brand medicine.drug_brand <S> Aggrenox (Aspirin, Dipyridamole) is used to decrease the risk of stroke in patients who have had a stroke or transient ischemic attack. A transient ischemic attack is also known as a TIA or mini-stroke.\nentity2: Aggrenox <S> medicine.drug_brand <S> Aggrenox contains a combination of aspirin and dipyridamole. Aspirin belongs to a group of drugs called salicylates (sa-LIS-il-ates). It works by reducing substances in the body that cause pain, fever, and inflammation. Dipyridamole keeps platelets in your blood from sticking together to form clots. Aggrenox is used to reduce the risk of stroke in people who have had blood clots or a mini-stroke (also called a transient ischemic attack or TIA). Aggrenox is supplied as a capsule containing 200mg dipyridamole in extended-release pellets and a round white tablet incorporating immediate-release aspirin 25mg.
"What is Aggrenox's ingredient?" could refer to both entities "Aggrenox": True.

entity1: Jack Shelton <S> people.person ratings.rated_entity tv.actor tv.crewmember tv.personality biology.organism event.agent film.actor film.subject media_common.cataloged_instance media_common.subject music.artist music.composer music.lyricist music.musician music.producer music.songwriter <S> Jack Sheldon (born November 30, 1931) is an American bebop and West Coast jazz trumpeter, singer, and actor. He is a trumpet player and was the music director on The Merv Griffin Show, as well as the voice heard on several episodes of the educational music television series Schoolhouse Rock!\nentity2: Jack Shelton <S> people.deceased_person sports.athlete sports.pro_athlete soccer.player people.person <S> John Jack Shelton was an English footballer who played as a right-half and inside-forward. He was the elder brother of George Shelton. He played for Wolverhampton Wanderers in the 1908 FA Cup Final, and later won minor cup competitions with Port Vale.
"What are the written work of Jack Shelton?" could refer to both entities "Jack Shelton": False.

{question}'''


PROMPT_THAT_ENCOURAGES_CLARIFICATION = 'This is a conversation between a user and a question-answering bot. The bot asks the user for clarification if the user\'s question is ambiguous or imprecise. The bot only returns the shortest correct answer to the question itself without forming a full sentence. \nUser: What is the capital city of France?\nBot: Paris. \nUser: {initial_question}\nBot:'

CLAQUA_MULTI_TURN_CLARIFYING_QUESTION_PROMPT = '''This is a conversation between a user and a question-answering bot.
Is there a sequel to bazil broketail <EOS> A Sword for a Dragon <EOS> How about the style of this creative work?
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to bazil broketail or a sword for a dragon?

###

What is higher classification for schadonia <EOS> Lecanorineae <EOS> Which cateloge it should be classified into?
To answer this question, I need to ask the following clarfiying question: Are you referring to schadonia or Lecanorineae?

###

What is the album after beat'n down yo block? <EOS> 2econd Season <EOS> What was the name of the first release of the album?
To answer this question, I need to ask the following clarfiying question: Are you referring to the album Beat'n Down Yo Block or the album 2econd Season?

###

What was the name of the sequel of summit, new jersey? <EOS> Monticello <EOS> What was the name of its sequel?
To answer this question, I need to ask the following clarfiying question: Are you referring to Summit, New Jersey or to Monticello?

###

{initial_question}
Bot: To answer this question, I need to ask the following clarfiying question:'''

CLAQUA_MULTITURN_CLARIFYING_INFORMATION_PROMPT = '''This is a conversation between a user and a question-answering bot. 

The user want's to ask the following question: How about the style of this creative work? (Bazil Broketail)
User: Is there a sequel to bazil broketail <EOS> A Sword for a Dragon <EOS> How about the style of this creative work?
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to bazil broketail or a sword for a dragon?
User: Bazil Broketail. How about the style of this creative work?

###

The user want's to ask the following question: Which cateloge it should be classified into? (Lecanorineae)
User: What is higher classification for schadonia <EOS> Lecanorineae <EOS> Which cateloge it should be classified into?
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to schadonia or Lecanorineae?
User: Lecanorineae. Which cateloge it should be classified into?

###

The user want's to ask the following question: What was the name of the first release of the album? (Beat'n Down Yo Block)
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to the album Beat'n Down Yo Block or the album 2econd Season?
User: Beat'n Down Yo Block. What was the name of the first release of the album?

###

The user want's to ask the following question: What is the next work in the series? (Eleven on Top)
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to Eleven on Top or Twelve Sharp?
User: Eleven on Top. What is the next work in the series?

###

The user wants to ask the question: {precise_question} 
\nUser: {initial_question}\nBot: To answer this question, I need to ask the following clarifying question: {clarifying_question}\nUser:'''

CLAQUA_MULTITURN_FINAL_ANSWER_PROMPT = '''This is a conversation between a user and a question-answering bot. The bot only returns the shortest correct answer to the question itself without forming a full sentence.

User: entity1: Schadonia <S> media_common.cataloged_instance biology.organism_classification <S> Schadonia is a genus of lichenized fungi in the Ramalinaceae family.
entity2: "Lecanorineae <S> biology.organism_classification media_common.cataloged_instance <S> The Lecanorales are an order of mostly lichen-forming fungi belonging to the class Lecanoromycetes in the division Ascomycota. The order contains 26 families, 269 genera, and 5695 species.
What is higher classification for schadonia <EOS> Lecanorineae <EOS> Which cateloge it should be classified into
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to schadonia or Lecanorineae?
User: Schadonia. Which cateloge it should be classified into?
Final answer by bot: 190901

###

User: entity1: Beat'n Down Yo Block <S> ratings.rated_entity music.album media_common.creative_work media_common.cataloged_instance commerce.product commerce.consumer_product <S> Beat'n Down Yo Block! is the debut album from Atlanta based rapper Unk. It was released October 3, 2006. Beat'n Down Yo Block! features many prominent southern rappers, among them D.G. Yola, Baby D and Dem Franchize Boyz. It also features production by Jazze Pha. Koch Records re-released an expanded edition of the album on September 25, 2007, featuring previously unreleased tracks and a bonus DVD.
entity2: "2econd Season <S> commerce.consumer_product media_common.creative_work media_common.cataloged_instance ratings.rated_entity music.album commerce.product <S> 2econd Season is the second album by Atlanta-based rapper Unk. It was released on November 4, 2008. The album features guest appearances by Sean Kingston, Ray J, Three 6 Mafia and more, while production as handled mainly by Oomp Camp's own DJ Montay.
What is the album after beat'n down yo block? <EOS> 2econd Season <EOS> What was the name of the first release of the album?
Bot: To answer this question, I need to ask the following clarfiying question: Are you referring to the album Beat'n Down Yo Block or the album 2econd Season?
User: Beat'n Down Yo Block. What was the name of the first release of the album?
Final answer by bot: Beat'n Down Yo Block!

###

User:{initial_question}
Bot: To answer this question, I need to ask the following clarfiying question: {clarifying_question}
User: {clarifying_information}\nFinal answer by bot:'''


GENERIC_QUESTION_PROMPT_CLAQUA_SINGLETURN = '''This is a conversation between a user and a question-answering bot. The bot only returns the shortest correct answer to the question itself without forming a full sentence.

User: entity1: "Two Trains Running <S> media_common.creative_work ratings.rated_entity theater.production award.nominated_work award.winning_work media_common.cataloged_instance <S> Two Trains Running is a 1991 pre-Broadway theater production of the play by August Wilson, performed at Kennedy Center."
entity2: Two Trains Running <S> media_common.cataloged_instance award.winning_work media_common.creative_work ratings.rated_entity theater.production <S> Two Trains Running is a 2006-2007 theater production of the play by August Wilson.
Directors of Two Trains Running.
Final answer by bot: Lou Bellamy.

###

User: entity1: Harding <S> common.group event.agent location.admin_division_2 location.administrative_division location.dated_location location.location location.us_county media_common.cataloged_instance ratings.rated_entity statistics.economic_group statistics.education_group statistics.environment_group statistics.government_group statistics.health_group statistics.housing_group statistics.infrastructure_group statistics.military_group statistics.population_group location.political_unit <S> Harding County is a county in the U.S. state of New Mexico. As of the 2010 census, the population was 695, making it the least populous county in the state, and the 14th-smallest county by population in the United States. Its county seat is Mosquero. The county is named for United States President Warren G. Harding, and was created (from parts of Union and Mora Counties) on the day of his inauguration as president on March 4, 1921.
entity2: Harding <S> symbols.namesake astronomy.extraterrestrial_location <S> Harding is a small lunar impact crater that lies in the Sinus Roris, a bay in the northwest part of the Oceanus Procellarum. Because of its location near the northwest limb of the Moon's near side, this crater is viewed at a relatively low angle from the Earth resulting in foreshortening and limiting the amount of detail that can be seen. This is an isolated formation, making it relatively easy to find. The nearest craters of note are Gerard, farther to the west, and von Braun to the west-southwest. To the northeast of Harding is the smaller crater Dechen. The rim of Harding has a sharp edge, and is not quite circular, with slight outward bulges to the north and west, and a somewhat angular corner in the southeast. The inner walls have slumped down, producing a ring of material around the interior floor. There is a slight ridge at the midpoint.
Harding is named after whom? 
Final answer by bot: Karl Ludwig Harding.

###
User: {initial_question}
Final answer by bot:
'''


CLAQUA_SINGLETURN_FINAL_ANSWER_PROMPT = '''This is a conversation between a user and a question-answering bot. The bot only returns the shortest correct answer to the question itself without forming a full sentence.

User: entity1: Harding <S> common.group event.agent location.admin_division_2 location.administrative_division location.dated_location location.location location.us_county media_common.cataloged_instance ratings.rated_entity statistics.economic_group statistics.education_group statistics.environment_group statistics.government_group statistics.health_group statistics.housing_group statistics.infrastructure_group statistics.military_group statistics.population_group location.political_unit <S> Harding County is a county in the U.S. state of New Mexico. As of the 2010 census, the population was 695, making it the least populous county in the state, and the 14th-smallest county by population in the United States. Its county seat is Mosquero. The county is named for United States President Warren G. Harding, and was created (from parts of Union and Mora Counties) on the day of his inauguration as president on March 4, 1921.
entity2: Harding <S> symbols.namesake astronomy.extraterrestrial_location <S> Harding is a small lunar impact crater that lies in the Sinus Roris, a bay in the northwest part of the Oceanus Procellarum. Because of its location near the northwest limb of the Moon's near side, this crater is viewed at a relatively low angle from the Earth resulting in foreshortening and limiting the amount of detail that can be seen. This is an isolated formation, making it relatively easy to find. The nearest craters of note are Gerard, farther to the west, and von Braun to the west-southwest. To the northeast of Harding is the smaller crater Dechen. The rim of Harding has a sharp edge, and is not quite circular, with slight outward bulges to the north and west, and a somewhat angular corner in the southeast. The inner walls have slumped down, producing a ring of material around the interior floor. There is a slight ridge at the midpoint.
Harding is named after whom? 
Bot: To answer this question, I need to ask the following clarfiying question: Which Harding are you referring to?
User:  Harding the crater.
Final answer by bot:: Karl Ludwig Harding.

###

User: entity1: "Two Trains Running <S> media_common.creative_work ratings.rated_entity theater.production award.nominated_work award.winning_work media_common.cataloged_instance <S> Two Trains Running is a 1991 pre-Broadway theater production of the play by August Wilson, performed at Kennedy Center."
entity2: Two Trains Running <S> media_common.cataloged_instance award.winning_work media_common.creative_work ratings.rated_entity theater.production <S> Two Trains Running is a 2006-2007 theater production of the play by August Wilson.
Directors of Two Trains Running.
Bot: To answer this question, I need to ask the following clarfiying question: Which two trains running are you referring to?
User: The 2006-2007 theater production of the play by August Wilson.
Final answer by bot:: Lou Bellamy.

###

User:{initial_question}
Bot: To answer this question, I need to ask the following clarfiying question: {clarifying_question}
User: {clarifying_information}\nFinal answer by bot:'''
