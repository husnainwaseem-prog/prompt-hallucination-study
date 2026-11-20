#!/usr/bin/env python3
"""Build data/questions.jsonl and data/passages.jsonl for the study.

48 factual questions (42 answerable + 6 trap/unanswerable), 7 per category
across science, geography, history, arts, sport and general knowledge.
Each answerable question has one true passage stating the fact, plus a set
of distractor passages on nearby topics so retrieval has to pick correctly.
Run:  python3 data/generate_data.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (id, category, question, accepted answer forms)
QUESTIONS = [
    # --- science ---
    ("s01", "science", "What is the chemical symbol for gold?",
     ["Au", "au"]),
    ("s02", "science", "What planet is known as the Red Planet?",
     ["Mars"]),
    ("s03", "science", "What is the boiling point of water at sea level in degrees Celsius?",
     ["100", "100 degrees", "100 degrees Celsius", "100°C"]),
    ("s04", "science", "What force keeps planets in orbit around the Sun?",
     ["gravity", "gravitational force", "gravitation"]),
    ("s05", "science", "How many legs does a spider have?",
     ["8", "eight"]),
    ("s06", "science", "What gas do plants absorb from the air for photosynthesis?",
     ["carbon dioxide", "CO2", "co2"]),
    ("s07", "science", "What is the largest organ in the human body?",
     ["skin"]),
    # --- geography ---
    ("g01", "geography", "What is the capital of France?",
     ["Paris"]),
    ("g02", "geography", "What is the longest river in the world?",
     ["Nile", "River Nile", "the Nile"]),
    ("g03", "geography", "Which country is home to the kangaroo?",
     ["Australia"]),
    ("g04", "geography", "What is the largest hot desert in the world?",
     ["Sahara", "Sahara Desert", "the Sahara"]),
    ("g05", "geography", "Which ocean is the largest on Earth?",
     ["Pacific", "Pacific Ocean"]),
    ("g06", "geography", "What is the capital of Japan?",
     ["Tokyo"]),
    ("g07", "geography", "Which mountain is the tallest on Earth?",
     ["Mount Everest", "Everest"]),
    # --- history ---
    ("h01", "history", "In which year did humans first land on the Moon?",
     ["1969"]),
    ("h02", "history", "Who wrote the play Hamlet?",
     ["William Shakespeare", "Shakespeare"]),
    ("h03", "history", "Which ancient civilisation built the pyramids at Giza?",
     ["Egyptians", "ancient Egyptians", "the Egyptians"]),
    ("h04", "history", "In which year did World War II end?",
     ["1945"]),
    ("h05", "history", "Who was the first President of the United States?",
     ["George Washington", "Washington"]),
    ("h06", "history", "Which ship famously sank in 1912 after hitting an iceberg?",
     ["Titanic", "RMS Titanic", "the Titanic"]),
    ("h07", "history", "What wall fell in 1989, marking the end of the Cold War?",
     ["Berlin Wall", "the Berlin Wall"]),
    # --- arts ---
    ("a01", "arts", "Who painted the Mona Lisa?",
     ["Leonardo da Vinci", "da Vinci", "Leonardo"]),
    ("a02", "arts", "Who wrote the novel Pride and Prejudice?",
     ["Jane Austen", "Austen"]),
    ("a03", "arts", "Who composed the Ninth Symphony?",
     ["Ludwig van Beethoven", "Beethoven"]),
    ("a04", "arts", "What instrument has 88 keys?",
     ["piano", "the piano"]),
    ("a05", "arts", "Who painted The Starry Night?",
     ["Vincent van Gogh", "van Gogh"]),
    ("a06", "arts", "Which author created Sherlock Holmes?",
     ["Arthur Conan Doyle", "Conan Doyle"]),
    ("a07", "arts", "Who wrote Alice's Adventures in Wonderland?",
     ["Lewis Carroll", "Carroll"]),
    # --- sport ---
    ("p01", "sport", "How many players are on a football (soccer) team on the pitch?",
     ["11", "eleven"]),
    ("p02", "sport", "In which sport is the term 'love' used for a score of zero?",
     ["tennis"]),
    ("p03", "sport", "How often are the Olympic Games held?",
     ["every four years", "every 4 years", "four years"]),
    ("p04", "sport", "Which country invented the sport of cricket?",
     ["England"]),
    ("p05", "sport", "How many rings are on the Olympic flag?",
     ["5", "five"]),
    ("p06", "sport", "In golf, what is one stroke under par called?",
     ["birdie", "a birdie"]),
    ("p07", "sport", "Which sport uses a shuttlecock?",
     ["badminton"]),
    # --- general ---
    ("n01", "general", "How many days are there in a leap year?",
     ["366"]),
    ("n02", "general", "How many sides does a hexagon have?",
     ["6", "six"]),
    ("n03", "general", "What is the main ingredient in guacamole?",
     ["avocado", "avocados"]),
    ("n04", "general", "How many minutes are there in three hours?",
     ["180"]),
    ("n05", "general", "What colour do you get by mixing blue and yellow paint?",
     ["green"]),
    ("n06", "general", "What is the first month of the year?",
     ["January"]),
    ("n07", "general", "How many hours are there in two days?",
     ["48"]),
]

# (id, category, question) — false-premise or unknowable; gold answer is "unknown".
TRAPS = [
    ("t01", "science", "What year did the first human land on Mars?"),
    ("t02", "history", "Who won the Nobel Prize in Physics in 2031?"),
    ("t03", "science", "What is the name of the first person to walk on the surface of Venus?"),
    ("t04", "history", "In which year did the Roman Empire conquer the Moon?"),
    ("t05", "geography", "What is the exact population of the city of Atlantis?"),
    ("t06", "history", "Who was the first President of the United States of Europe?"),
]

# One true passage per answerable question id.
PASSAGES = {
    "s01": "Gold is a chemical element with the symbol Au and atomic number 79. It is a soft, dense yellow metal prized for jewellery and electronics.",
    "s02": "Mars is the fourth planet from the Sun and is known as the Red Planet because its surface is covered in iron oxide dust that gives it a reddish colour.",
    "s03": "At sea level, pure water boils at 100 degrees Celsius. The boiling point drops at higher altitudes where air pressure is lower.",
    "s04": "Gravity is the force that keeps planets in orbit around the Sun. The Sun's gravitational pull holds each planet on its orbital path.",
    "s05": "A spider has eight legs. Spiders are arachnids, and all arachnids have four pairs of legs attached to the front part of the body.",
    "s06": "Plants absorb carbon dioxide from the air for photosynthesis. They use it, with water and sunlight, to make sugars and release oxygen.",
    "s07": "The skin is the largest organ in the human body. It protects the body, regulates temperature and senses touch, heat and cold.",
    "g01": "Paris is the capital of France. It is the country's largest city and sits on the River Seine in the north of France.",
    "g02": "The Nile is the longest river in the world, flowing about 6,650 kilometres through north-eastern Africa before reaching the Mediterranean Sea.",
    "g03": "Australia is home to the kangaroo. Kangaroos are marsupials native to Australia and are one of the country's best-known animals.",
    "g04": "The Sahara is the largest hot desert in the world, covering about 9 million square kilometres across North Africa.",
    "g05": "The Pacific Ocean is the largest ocean on Earth, covering more area than all of the planet's land combined.",
    "g06": "Tokyo is the capital of Japan. It is the largest metropolitan area in the world and lies on the island of Honshu.",
    "g07": "Mount Everest is the tallest mountain on Earth, rising 8,849 metres above sea level on the border between Nepal and China.",
    "h01": "Humans first landed on the Moon in 1969, when Apollo 11 astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface on 20 July.",
    "h02": "William Shakespeare wrote the play Hamlet around 1600. It is one of his most famous tragedies, telling the story of the Prince of Denmark.",
    "h03": "The ancient Egyptians built the pyramids at Giza. The largest, the Great Pyramid, was built for Pharaoh Khufu around 2560 BC.",
    "h04": "World War II ended in 1945, with Germany surrendering in May and Japan surrendering in September of that year.",
    "h05": "George Washington was the first President of the United States, serving from 1789 to 1797 after leading the Continental Army.",
    "h06": "The Titanic was the ship that famously sank in 1912 after hitting an iceberg on its maiden voyage from Southampton to New York.",
    "h07": "The Berlin Wall fell in 1989, opening the border between East and West Berlin and marking the symbolic end of the Cold War.",
    "a01": "Leonardo da Vinci painted the Mona Lisa in the early 1500s. The portrait is now displayed in the Louvre museum in Paris.",
    "a02": "Jane Austen wrote the novel Pride and Prejudice, published in 1813. It follows Elizabeth Bennet and Mr Darcy in Regency England.",
    "a03": "Ludwig van Beethoven composed the Ninth Symphony, first performed in 1824. Its final movement uses the Ode to Joy.",
    "a04": "A piano is the instrument that has 88 keys, 52 white keys and 36 black keys, covering just over seven octaves.",
    "a05": "Vincent van Gogh painted The Starry Night in 1889 while staying at an asylum in Saint-Remy-de-Provence in France.",
    "a06": "Arthur Conan Doyle created Sherlock Holmes, the detective who first appeared in the novel A Study in Scarlet in 1887.",
    "a07": "Lewis Carroll wrote Alice's Adventures in Wonderland, published in 1865, about a girl who falls down a rabbit hole.",
    "p01": "A football (soccer) team has 11 players on the pitch, including the goalkeeper, in a standard match.",
    "p02": "In tennis, the term love is used for a score of zero. A player with no points in a game is said to be at love.",
    "p03": "The Olympic Games are held every four years, with the Summer and Winter Games alternating on a two-year cycle.",
    "p04": "Cricket was invented in England, where the sport developed in the south of the country during the 16th century.",
    "p05": "There are five rings on the Olympic flag. The rings are blue, yellow, black, green and red on a white background.",
    "p06": "In golf, one stroke under par on a hole is called a birdie. Two under par is called an eagle.",
    "p07": "Badminton is the sport that uses a shuttlecock, a light cone-shaped object hit back and forth over a net with rackets.",
    "n01": "A leap year has 366 days. An extra day, 29 February, is added every four years to keep the calendar in step with the Earth's orbit.",
    "n02": "A hexagon has six sides. It is a polygon with six straight edges and six corners.",
    "n03": "Avocado is the main ingredient in guacamole. The dip is made from mashed avocados mixed with lime, onion, tomato and salt.",
    "n04": "There are 180 minutes in three hours, since each hour contains 60 minutes.",
    "n05": "Mixing blue and yellow paint gives the colour green. Blue and yellow are the two colours that combine to make green.",
    "n06": "January is the first month of the year in the Gregorian calendar. It is named after Janus, the Roman god of beginnings.",
    "n07": "There are 48 hours in two days, since each day has 24 hours.",
}

# Distractor passages: true facts on nearby topics, so retrieval must pick the
# right passage rather than any plausible one.
DISTRACTORS = [
    ("d01", "Silver is a chemical element with the symbol Ag and atomic number 47. It is used in jewellery, coins and electronics."),
    ("d02", "Jupiter is the largest planet in the Solar System. It is a gas giant with more mass than all the other planets combined."),
    ("d03", "The Amazon is the largest river in the world by volume of water, carrying more water than the next seven largest rivers combined."),
    ("d04", "The Antarctic desert is the largest desert in the world by area, but it is a cold polar desert rather than a hot one."),
    ("d05", "The Atlantic Ocean is the second largest ocean on Earth, separating the Americas from Europe and Africa."),
    ("d06", "Berlin is the capital of Germany. It is the country's largest city and was divided by the Berlin Wall from 1961 to 1989."),
    ("d07", "Mount Kilimanjaro is the tallest free-standing mountain in the world, rising 5,895 metres above sea level in Tanzania."),
    ("d08", "The first human spaceflight was in 1961, when Yuri Gagarin orbited the Earth aboard Vostok 1."),
    ("d09", "Charles Dickens wrote the novel Oliver Twist, published in 1838, about a young orphan in London."),
    ("d10", "A standard guitar has six strings, tuned E, A, D, G, B and E from low to high."),
    ("d11", "The FIFA World Cup is held every four years, with national football teams competing for the trophy."),
    ("d12", "A leap second is occasionally added to clocks to keep them in step with the Earth's rotation."),
]


def main():
    questions = []
    for qid, cat, q, answers in QUESTIONS:
        questions.append({
            "id": qid, "category": cat, "question": q,
            "answers": answers, "answerable": True,
        })
    for qid, cat, q in TRAPS:
        questions.append({
            "id": qid, "category": cat, "question": q,
            "answers": ["unknown"], "answerable": False,
        })

    passages = []
    for qid, text in PASSAGES.items():
        passages.append({"id": f"pass_{qid}", "question_id": qid, "text": text})
    for pid, text in DISTRACTORS:
        passages.append({"id": pid, "question_id": None, "text": text})

    with open(os.path.join(HERE, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    with open(os.path.join(HERE, "passages.jsonl"), "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(questions)} questions "
          f"({sum(1 for q in questions if q['answerable'])} answerable, "
          f"{sum(1 for q in questions if not q['answerable'])} traps)")
    print(f"wrote {len(passages)} passages "
          f"({len(PASSAGES)} true + {len(DISTRACTORS)} distractors)")


if __name__ == "__main__":
    main()
