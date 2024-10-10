#ifndef THIRD_PARTY_GEMMA_CPP_EVALS_PROMPTS_H_
#define THIRD_PARTY_GEMMA_CPP_EVALS_PROMPTS_H_

#include "hwy/base.h"

// The input prompts, each named by its token length.
static const char* Prompt32() {
  return "In the heart of a bustling marketplace, amidst the vibrant colors "
         "and lively chatter, there existed a realm where fruits reigned "
         "supreme. Each fruit, with its unique shape, texture, and flavor, "
         "held a story waiting to be told.";
}

static const char* Prompt64() {
  return "There was the regal Apple, its skin a vibrant red, its flesh a crisp "
         "white. It was a symbol of knowledge and temptation, reminding all of "
         "the Garden of Eden and the pursuit of wisdom. The Apple, with its "
         "sweet and slightly tart taste, was a versatile fruit, enjoyed on its "
         "own, baked into pies, or transformed into cider. ";
}

static const char* Prompt128() {
  return "Beside the Apple stood the cheerful Banana, its curved yellow form a "
         "beacon of sunshine. The Banana, with its creamy texture and sweet "
         "flavor, was a source of energy and happiness. It was a fruit that "
         "brought smiles to faces, whether eaten as a snack, blended into "
         "smoothies, or used to create delicious banana bread. Across the way, "
         "the playful Grapes hung in clusters, their translucent green and "
         "purple orbs glistening like jewels. The Grapes, with their juicy "
         "flesh and refreshing taste, were a symbol of abundance and "
         "celebration. They were a fruit that added a touch of elegance to any "
         "occasion, whether";
}

static const char* Prompt256() {
  return "Next to the Grapes, the prickly Pineapple stood tall, its spiky "
         "exterior concealing a sweet and tangy treasure. The Pineapple, with "
         "its golden flesh and tropical aroma, was a symbol of hospitality and "
         "warmth. It was a fruit that transported taste buds to faraway lands, "
         "whether enjoyed on its own, grilled, or used to create refreshing "
         "cocktails. And in the corner, the humble Orange shone brightly, its "
         "citrusy scent filling the air. The Orange, with its juicy segments "
         "and tangy flavor, was a symbol of vitality and health. It was a "
         "fruit that invigorated the senses, whether enjoyed as a snack, "
         "squeezed into juice, or used to create zesty marmalades. ";
}

static const char* Prompt512() {
  return "In the marketplace, the fruits coexisted harmoniously, each "
         "contributing its own unique essence to the vibrant tapestry of "
         "flavors. They were a reminder that diversity is beautiful, and that "
         "every fruit, no matter how big or small, has a story to tell. As the "
         "day progressed, people from all walks of life flocked to the "
         "marketplace, drawn by the allure of the fruits. There was the young "
         "child, eyes wide with wonder, reaching for a plump strawberry, its "
         "bright red hue promising a burst of sweetness. There was the elderly "
         "couple, sharing a juicy mango, its golden flesh evoking memories of "
         "their youth. And there was the chef, carefully selecting a variety "
         "of fruits, their vibrant colors and textures inspiring culinary "
         "creations that would tantalize taste buds. Among the crowd, a "
         "storyteller captivated listeners with tales of the fruits' origins "
         "and symbolism. He spoke of the pomegranate, its ruby-red seeds "
         "representing fertility and abundance, and the fig, its sweetness "
         "signifying peace and prosperity. He told of the watermelon, its "
         "refreshing juice quenching thirst on hot summer days, and the kiwi, "
         "its vibrant green flesh offering a taste of the exotic. ...As the "
         "sun began its descent, casting long shadows across the marketplace, "
         "a hush fell over the crowd.  A young girl, no older than seven, with "
         "bright eyes and a mischievous grin, approached the storyteller.  "
         "Clutching a small, bruised apple in her hand, she asked, \"What "
         "about my apple?  It's not pretty like the others.  Does it have a "
         "story too?\" The storyteller smiled warmly, taking the apple gently "
         "from her hand.  \"Ah,\" he began, his voice resonating through the "
         "square, \"this apple, though seemingly imperfect, holds a story of "
         "resilience and transformation.  You see, where others may see a "
         "blemish, we can see a mark of character, a reminder that true beauty "
         "lies within. He held the apple aloft for all to see.  \"This "
         "apple,\" he continued, \"has faced the challenges of nature – the "
         "wind, the rain, perhaps even a hungry bird.  But it has endured, its "
         "sweetness preserved, its essence intact.  It reminds us that even in "
         "the face of adversity, we can emerge stronger, our spirits "
         "unbroken.\"The girl's eyes widened, her smile returning brighter "
         "than before.  Inspired by the storyteller's words, she took a bite "
         "of her apple, savoring its sweet, slightly tart flavor.  It was the "
         "most delicious apple she had";
}

static HWY_MAYBE_UNUSED const char* GetPrompt(int length) {
  switch (length) {
    case 32:
      return Prompt32();
    case 64:
      return Prompt64();
    case 128:
      return Prompt128();
    case 256:
      return Prompt256();
    case 512:
      return Prompt512();
    default:
      return Prompt32();
  }
}

#endif  // THIRD_PARTY_GEMMA_CPP_EVALS_PROMPTS_H_
