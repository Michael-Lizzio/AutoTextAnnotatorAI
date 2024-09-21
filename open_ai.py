from openai import OpenAI
import json
import re
from secret import OPENAI_API_KEY


class Annotate:
    def __init__(self, api_key=OPENAI_API_KEY):
        # Initialize the OpenAI client with API key
        self.client = OpenAI(api_key=api_key)

    def clean_text(self, text):
        """Remove non-alphanumeric characters from the text."""
        # This regex replaces anything that's not a letter, number, or space with nothing
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        return text.strip()

    def make_response_json(self, response):
        """Converts response annotations into a clean JSON object."""
        # Extracting the annotation content from the response
        # Splitting the raw content into separate annotations
        raw_annotations = response.split('\n')

        # Cleaning and structuring the annotations
        # print(response)
        json_annotations = {}
        for annotation in raw_annotations:
            if annotation:  # Ensuring the annotation is not empty
                outcome = annotation.split('-')
                print(outcome)
                if len(outcome) == 2:
                    key, value = outcome
                else:
                    continue

                clean_key = self.clean_text(key)
                clean_value = self.clean_text(value)
                json_annotations[clean_key] = clean_value
        return json_annotations

    def annotate_text_with_json_mode(self, text, prompt, attempt=0):
        """
        Annotates a given text according to a specific prompt using OpenAI's Chat API in JSON mode.

        Parameters:
            text (str): The text to annotate.
            prompt (str): The prompt to guide the annotation.

        Returns:
            str: The annotated text in a structured JSON format.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant designed to output annotations in JSON. Please follow these instructions: " + prompt},
                {"role": "user", "content": text}
            ]
        )

        # Extracting and printing the annotation content from the response
        json_annotations = {}
        raw_content = response.choices[0].message.content
        try:
            json_annotations = self.make_response_json(raw_content)
        except Exception as ex:
            print("Error", attempt, ex)
            if attempt < 2:
                self.annotate_text_with_json_mode(text, prompt, attempt=attempt + 1)
        if len(json_annotations) == 0:
            if attempt < 2:
                self.annotate_text_with_json_mode(text, prompt, attempt=attempt + 1)
        return json_annotations

    def get_annotations(self, text_to_annotate, word_length=5, annotations_per_page=5,
                        example="Phrase/Search words - Annotation", extra=""):
        # Example text and prompt
        prompt = f"""Please take annotations on this page of writing. {extra}The annotations should be no more than {word_length} words long and 
        there should be at most {annotations_per_page} on a page. When writing the annotations you should reply with the first 7 words of the: 
        phrase, quote, sentence, paragraph. Then next to that have the annotation. make sure to evenly add annotations.
        \nExample ({example}\n{example})\nHere is the text."""

        # Call the function and print the output
        return self.annotate_text_with_json_mode(text_to_annotate, prompt)


if __name__ == '__main__':
    text_to_annotate = """A green and yellow parrot, which hung in a cage outside the door, kept repeating over and over: "Allez vous-en! Allez vous-en! Sapristi! That's all right!" He could speak a little Spanish, and also a language which nobody understood, unless it was the mocking-bird that hung on the other side of the door, whistling his fluty notes out upon the breeze with maddening persistence. Mr. Pontellier, unable to read his newspaper with any degree of comfort, arose with an expression and an exclamation of disgust. He walked down the gallery and across the narrow "bridges" which connected the Lebrun cottages one with the other. He had been seated before the door of the main house. The parrot and the mockingbird were the property of Madame Lebrun, and they had the right to make all the noise they wished. Mr. Pontellier had the privilege of quitting their society when they ceased to be entertaining. He stopped before the door of his own cottage, which was the fourth one from the main building and next to the last. Seating himself in a wicker rocker which was there, he once more applied himself to the task of reading the newspaper. The day was Sunday; the paper was a day old. The Sunday papers had not yet reached Grand Isle. He was already acquainted with the market reports, and he glanced restlessly over the editorials and bits of news which he had not had time to read before quitting New Orleans the day before. Mr. Pontellier wore eye-glasses. He was a man of forty, of medium height and rather slender build; he stooped a little. His hair was brown and straight, parted on one side. His beard was neatly and closely trimmed. Once in a while he withdrew his glance from the newspaper and looked about him. There was more noise than ever over at the house. The main building was called "the house,' to distinguish it from the cottages. The chattering and whistling birds were still at it. Two young girls, the Farival twins, were playing a duet from "Zampa" upon the piano. Madame Lebrun was bustling in and out, giving orders in a high key to a yard-boy whenever she got inside the house, and directions in an equally high voice to a dining-room servant whenever she got outside. She was a fresh, pretty woman, clad always in white with elbow sleeves. Her starched skirts crinkled as she came and went. Farther down, before one of the cottages, a lady in black was walking demurely up and down, telling her beads. A good many persons of the pension had gone over to the Cheniere Caminada in Beaudelet's lugger to hear mass. Some young people were out under the wateroaks playing croquet. Mr. Pontellier's two children were there sturdy little fellows of four and five. A quadroon nurse followed them about with a faraway, meditative air. Mr. Pontellier finally lit a cigar and began to smoke, letting the paper drag idly from his hand. He fixed his gaze upon a white sunshade that was advancing at snail's pace from the beach. He could see it plainly between the gaunt trunks of the water-oaks and across the stretch of yellow camomile. The gulf looked far away, melting hazily into the blue of the horizon. The sunshade continued to approach slowly. Beneath its pink-lined shelter were his wife, Mrs. Pontellier, and young Robert Lebrun. When they reached the cottage, the two seated themselves with some appearance of fatigue upon the upper step of the porch, facing each other, each leaning against a supporting post."""
    a = Annotate()
    print(a.get_annotations(text_to_annotate))
