#ifndef PROMPTS_H  // Include guard to prevent multiple inclusions
#define PROMPTS_H

// Prompts for different tasks
static const char* ShortPrompt() { return "What is the capital of Spain?"; }

static const char* FactualityPrompt() {
  return "How does an inkjet printer work?";
}

static const char* CreativePrompt() {
  return "Tell me a story about a magical bunny and their TRS-80.";
}

static const char* CodingPrompt() {
  return "Write a python program to generate a fibonacci sequence.";
}

// The input prompts, each named by its token length.

static const char* Prompt32() {
  return "Once upon a time, there existed a little girl who liked to have "
         "adventures. She wanted to go to places and";
}

static const char* Prompt64() {
  return "It is done, and submitted. You can play 'Survival of the Tastiest' "
         "on Android, and on the web. Playing on the web works, but you have "
         "to simulate multiple touch for table moving and that can be a bit "
         "confusing. There is a lot ";
}

static const char* Prompt128() {
  return "It's done, and submitted. You can play 'Survival of the Tastiest' on "
         "Android, and on the web. Playing on the web works, but you have to "
         "simulate multiple touch for table moving and that can be a bit "
         "confusing. There is a lot I'd like to talk about. I will go through"
         " every topic, insted of making the typical what went right/wrong list"
         ". Concept Working over the theme was probably one of the hardest "
         "tasks which I had to face. Originally, I had an idea of what kind of "
          "game I wanted to develop, gamep";
}

static const char* Prompt256() {
  return "It is done, and submitted. You can play 'Survival of the Tastiest' on"
         " Android, and on the web. Playing on the web works, but you have to "
         "simulate multiple touch for table moving and that can be a bit "
         "confusing. There is a lot I'd like to talk about. I will go through "
         "every topic, insted of making the typical what went right/wrong list."
         " Concept Working over the theme was probably one of the hardest tasks"
         " which I had to face. Originally, I had an idea of what kind of game "
         "I wanted to develop, gameplay wise - something with a lot of "
         "enemies/actors, simple graphics, maybe set in space, controlled from "
         "a top-down view. I was confident that I could fit any theme around "
         "it. In the end, the problem with a theme like 'Evolution' in a game "
         "is that evolution is unassisted. It happens through several seemingly"
         " random mutations over time, with the most apt permutation surviving."
         " This genetic car simulator is, in my opinion, a great example of "
         "actual evolution of a species facing a challenge. But is it a game? "
         "In a game, you need to control something to reach an objective. This "
         "could be a character, a ";
}

static const char* Prompt512() {
  return  "It is done, and submitted. You can play 'Survival of the Tastiest'"
    " on Android, and on the web. Playing on the web works, but you have to"
    " simulate multiple touch for table moving and that can be a bit"
    " confusing. There is a lot I'd like to talk about. I will go through"
    " every topic, instead of making the typical what went right/wrong list."
    " Concept Working over the theme was probably one of the hardest tasks"
    " which I had to face. Originally, I had an idea of what kind of game I"
    " wanted to develop, gameplay wise - something with a lot of"
    " enemies/actors, simple graphics, maybe set in space, controlled from"
    " a top-down view. I was confident that I could fit any theme around"
    " it. In the end, the problem with a theme like 'Evolution' in a game"
    " is that evolution is unassisted. It happens through several seemingly"
    " random mutations over time, with the most apt permutation surviving."
    " This genetic car simulator is, in my opinion, a great example of"
    " actual evolution of a species facing a challenge. But is it a game?"
    " In a game, you need to control something to reach an objective. That"
    " control goes against what evolution is supposed to be like. If you"
    " allow the user to pick how to evolve something, it's not evolution"
    " anymore - it's the equivalent of intelligent design, the fable"
    " invented by creationists to combat the idea of evolution. Being"
    " agnostic and a Pastafarian, that's not something that rubbed me the"
    " right way. Hence, my biggest dillema when deciding what to create was"
    " not with what I wanted to create, but with what I did not. I didn't"
    " want to create an 'intelligent design' simulator and wrongly call it"
    " evolution. This is a problem, of course, every other contestant also"
    " had to face. And judging by the entries submitted, not many managed"
    " to work around it. I'd say the only real solution was through the use"
    " of artificial selection, somehow. So far, I have not seen any entry"
    " using this at its core gameplay. Alas, this is just a fun competition"
    " and after a while I decided not to be as strict with the game idea,"
    " and allowed myself to pick whatever I thought would work out. My"
    " initial idea was to create something where humanity tried to evolve"
    " to a next level but had some kind of foe trying to stop them from"
    " doing so. I kind of had this image of human souls flying in space"
    " towards ";
}

static const char* GetPrompt(int length) {
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
      return ShortPrompt();
  }
}

#endif
