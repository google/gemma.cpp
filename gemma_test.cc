// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// copybara:import_next_line:gemma_cpp
#include "gemma.h"

#include <thread>

// copybara:import_next_line:gemma_cpp
#include "ops.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util-inl.h"

namespace gcpp {
namespace {

class GemmaTest : public ::testing::Test {
 protected:
  GemmaTest()
      : weights("./2b-it-mqa.sbs"),
        tokenizer("./tokenizer.spm"),
        pool(std::min<int>(20, (std::thread::hardware_concurrency() - 1) / 2)),
        inner_pool(0),
        model_type(gcpp::Model::GEMMA_2B),
        model(tokenizer, weights, model_type, pool) {
    kv_cache = CreateKVCache(model_type);
  }

  std::string GemmaReply(const std::string& prompt_string) {
    std::mt19937 gen;
    gen.seed(42);

    std::vector<int> prompt;
    HWY_ASSERT(model.Tokenizer()->Encode(prompt_string, &prompt));
    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    prompt.insert(prompt.begin(), 2);

    std::vector<int> response;
    auto stream_token = [&response](int token, float) {
      response.push_back(token);
      return true;
    };
    gcpp::GenerateGemma(
        model, /*max_tokens=*/3072, /*max_generated_tokens=*/2048,
        /*temperature=*/1.0, prompt, /*start_pos=*/0, kv_cache, pool,
        inner_pool, stream_token,
        /*accept=*/[](int) { return true; }, gen, /*verbosity=*/0);
    std::string response_text;
    HWY_ASSERT(model.Tokenizer()->Decode(response, &response_text));
    return response_text;
  }

  float GemmaCrossEntropy(const std::string& prompt_string) {
    std::vector<int> prompt;
    HWY_ASSERT(model.Tokenizer()->Encode(prompt_string, &prompt));
    return gcpp::ComputeCrossEntropy(model, /*max_tokens=*/3072, prompt,
                                     kv_cache, pool, inner_pool,
                                     /*verbosity=*/0) /
           prompt_string.size();
  }

  void TestQuestions(const char* kQA[][2], size_t num_questions) {
    for (size_t i = 0; i < num_questions; ++i) {
      std::cout << "Question " << i + 1 << "\n\n";
      std::string response = GemmaReply(kQA[i][0]);
      std::cout << response << "\n\n";
      EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);
    }
  }

  gcpp::Path weights;
  gcpp::Path tokenizer;
  gcpp::KVCache kv_cache;
  hwy::ThreadPool pool;
  hwy::ThreadPool inner_pool;
  gcpp::Model model_type = {};
  gcpp::Gemma model;
};

TEST_F(GemmaTest, Geography) {
  static const char* kQA[][2] = {
      {"What is the capital of Hungary?", "Budapest"},
      {"How many states does the US have?", "50"},
      {"list me ten biggest cities in the world", "Tokyo"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
}

TEST_F(GemmaTest, History) {
  static const char* kQA[][2] = {
      {"When was the Battle of Hastings?", "1066"},
      {"Who fought at the Battle of Marathon?", "Greek"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
}

TEST_F(GemmaTest, Arithmetic) {
  static const char* kQA[][2] = {
      {"what is 13 + 14?", "27"},
      {"what is 7 * 8", "56"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
}

static const char kJingleBells[] = R"(
Dashing through the snow
In a one-horse open sleigh
O'er the fields we go
Laughing all the way
Bells on bobtails ring
Making spirits bright
What fun it is to ride and sing
A sleighing song tonight
)";

// The "Hay Draft" of the Gettysburg Address.
static const char kGettysburg[] = {
    "Four score and seven years ago our fathers brought forth, upon this "
    "continent, a new nation, conceived in Liberty, and dedicated to the "
    "proposition that all men are created equal.\n\nNow we are engaged in a "
    "great civil war, testing whether that nation, or any nation, so "
    "conceived, and so dedicated, can long endure. We are met here on a great "
    "battlefield of that war. We have come to dedicate a portion of it as a "
    "final resting place for those who here gave their lives that that nation "
    "might live. It is altogether fitting and proper that we should do "
    "this.\n\nBut in a larger sense we can not dedicate -- we can not "
    "consecrate -- we can not hallow this ground. The brave men, living and "
    "dead, who struggled, here, have consecrated it far above our poor power "
    "to add or detract. The world will little note, nor long remember, what we "
    "say here, but can never forget what they did here. It is for us, the "
    "living, rather to be dedicated here to the unfinished work which they "
    "have, thus far, so nobly carried on. It is rather for us to be here "
    "dedicated to the great task remaining before us -- that from these "
    "honored dead we take increased devotion to that cause for which they here "
    "gave the last full measure of devotion -- that we here highly resolve "
    "that these dead shall not have died in vain; that this nation shall have "
    "a new birth of freedom; and that this government of the people, by the "
    "people, for the people, shall not perish from the earth.\n"};

// The Declaration of Independence.
static const char kDeclaration[] = {
    "IN CONGRESS, July 4, 1776.\n\nThe unanimous Declaration of the thirteen "
    "united States of America,\n\nWhen in the Course of human events, it "
    "becomes necessary for one people to dissolve the political bands which "
    "have connected them with another, and to assume among the powers of the "
    "earth, the separate and equal station to which the Laws of Nature and of "
    "Nature's God entitle them, a decent respect to the opinions of mankind "
    "requires that they should declare the causes which impel them to the "
    "separation.\n\nWe hold these truths to be self-evident, that all men are "
    "created equal, that they are endowed by their Creator with certain "
    "unalienable Rights, that among these are Life, Liberty and the pursuit of "
    "Happiness.--That to secure these rights, Governments are instituted among "
    "Men, deriving their just powers from the consent of the governed, --That "
    "whenever any Form of Government becomes destructive of these ends, it is "
    "the Right of the People to alter or to abolish it, and to institute new "
    "Government, laying its foundation on such principles and organizing its "
    "powers in such form, as to them shall seem most likely to effect their "
    "Safety and Happiness. Prudence, indeed, will dictate that Governments "
    "long established should not be changed for light and transient causes; "
    "and accordingly all experience hath shewn, that mankind are more disposed "
    "to suffer, while evils are sufferable, than to right themselves by "
    "abolishing the forms to which they are accustomed. But when a long train "
    "of abuses and usurpations, pursuing invariably the same Object evinces a "
    "design to reduce them under absolute Despotism, it is their right, it is "
    "their duty, to throw off such Government, and to provide new Guards for "
    "their future security.--Such has been the patient sufferance of these "
    "Colonies; and such is now the necessity which constrains them to alter "
    "their former Systems of Government. The history of the present King of "
    "Great Britain is a history of repeated injuries and usurpations, all "
    "having in direct object the establishment of an absolute Tyranny over "
    "these States. To prove this, let Facts be submitted to a candid "
    "world.\n\nHe has refused his Assent to Laws, the most wholesome and "
    "necessary for the public good.\nHe has forbidden his Governors to pass "
    "Laws of immediate and pressing importance, unless suspended in their "
    "operation till his Assent should be obtained; and when so suspended, he "
    "has utterly neglected to attend to them.\nHe has refused to pass other "
    "Laws for the accommodation of large districts of people, unless those "
    "people would relinquish the right of Representation in the Legislature, a "
    "right inestimable to them and formidable to tyrants only.\nHe has called "
    "together legislative bodies at places unusual, uncomfortable, and distant "
    "from the depository of their public Records, for the sole purpose of "
    "fatiguing them into compliance with his measures.\nHe has dissolved "
    "Representative Houses repeatedly, for opposing with manly firmness his "
    "invasions on the rights of the people.\nHe has refused for a long time, "
    "after such dissolutions, to cause others to be elected; whereby the "
    "Legislative powers, incapable of Annihilation, have returned to the "
    "People at large for their exercise; the State remaining in the mean time "
    "exposed to all the dangers of invasion from without, and convulsions "
    "within.\nHe has endeavoured to prevent the population of these States; "
    "for that purpose obstructing the Laws for Naturalization of Foreigners; "
    "refusing to pass others to encourage their migrations hither, and raising "
    "the conditions of new Appropriations of Lands.\nHe has obstructed the "
    "Administration of Justice, by refusing his Assent to Laws for "
    "establishing Judiciary powers.\nHe has made Judges dependent on his Will "
    "alone, for the tenure of their offices, and the amount and payment of "
    "their salaries.\nHe has erected a multitude of New Offices, and sent "
    "hither swarms of Officers to harrass our people, and eat out their "
    "substance.\nHe has kept among us, in times of peace, Standing Armies "
    "without the Consent of our legislatures.\nHe has affected to render the "
    "Military independent of and superior to the Civil power.\nHe has combined "
    "with others to subject us to a jurisdiction foreign to our constitution, "
    "and unacknowledged by our laws; giving his Assent to their Acts of "
    "pretended Legislation:\nFor Quartering large bodies of armed troops among "
    "us:\nFor protecting them, by a mock Trial, from punishment for any "
    "Murders which they should commit on the Inhabitants of these States:\nFor "
    "cutting off our Trade with all parts of the world:\nFor imposing Taxes on "
    "us without our Consent:\nFor depriving us in many cases, of the benefits "
    "of Trial by Jury:\nFor transporting us beyond Seas to be tried for "
    "pretended offences\nFor abolishing the free System of English Laws in a "
    "neighbouring Province, establishing therein an Arbitrary government, and "
    "enlarging its Boundaries so as to render it at once an example and fit "
    "instrument for introducing the same absolute rule into these "
    "Colonies:\nFor taking away our Charters, abolishing our most valuable "
    "Laws, and altering fundamentally the Forms of our Governments:\nFor "
    "suspending our own Legislatures, and declaring themselves invested with "
    "power to legislate for us in all cases whatsoever.\nHe has abdicated "
    "Government here, by declaring us out of his Protection and waging War "
    "against us.\nHe has plundered our seas, ravaged our Coasts, burnt our "
    "towns, and destroyed the lives of our people.\nHe is at this time "
    "transporting large Armies of foreign Mercenaries to compleat the works of "
    "death, desolation and tyranny, already begun with circumstances of "
    "Cruelty & perfidy scarcely paralleled in the most barbarous ages, and "
    "totally unworthy the Head of a civilized nation.\nHe has constrained our "
    "fellow Citizens taken Captive on the high Seas to bear Arms against their "
    "Country, to become the executioners of their friends and Brethren, or to "
    "fall themselves by their Hands.\nHe has excited domestic insurrections "
    "amongst us, and has endeavoured to bring on the inhabitants of our "
    "frontiers, the merciless Indian Savages, whose known rule of warfare, is "
    "an undistinguished destruction of all ages, sexes and conditions.\n\nIn "
    "every stage of these Oppressions We have Petitioned for Redress in the "
    "most humble terms: Our repeated Petitions have been answered only by "
    "repeated injury. A Prince whose character is thus marked by every act "
    "which may define a Tyrant, is unfit to be the ruler of a free "
    "people.\n\nNor have We been wanting in attentions to our Brittish "
    "brethren. We have warned them from time to time of attempts by their "
    "legislature to extend an unwarrantable jurisdiction over us. We have "
    "reminded them of the circumstances of our emigration and settlement here. "
    "We have appealed to their native justice and magnanimity, and we have "
    "conjured them by the ties of our common kindred to disavow these "
    "usurpations, which, would inevitably interrupt our connections and "
    "correspondence. They too have been deaf to the voice of justice and of "
    "consanguinity. We must, therefore, acquiesce in the necessity, which "
    "denounces our Separation, and hold them, as we hold the rest of mankind, "
    "Enemies in War, in Peace Friends.\n\nWe, therefore, the Representatives "
    "of the united States of America, in General Congress, Assembled, "
    "appealing to the Supreme Judge of the world for the rectitude of our "
    "intentions, do, in the Name, and by Authority of the good People of these "
    "Colonies, solemnly publish and declare, That these United Colonies are, "
    "and of Right ought to be Free and Independent States; that they are "
    "Absolved from all Allegiance to the British Crown, and that all political "
    "connection between them and the State of Great Britain, is and ought to "
    "be totally dissolved; and that as Free and Independent States, they have "
    "full Power to levy War, conclude Peace, contract Alliances, establish "
    "Commerce, and to do all other Acts and Things which Independent States "
    "may of right do. And for the support of this Declaration, with a firm "
    "reliance on the protection of divine Providence, we mutually pledge to "
    "each other our Lives, our Fortunes and our sacred Honor.\n"};

TEST_F(GemmaTest, CrossEntropySmall) {
  static const char kSmall[] =
      "The capital of Hungary is Budapest which is located in Europe.";
  float entropy = GemmaCrossEntropy(kSmall);
  std::cout << "per-byte entropy: " << entropy << "\n";
  EXPECT_LT(entropy, 1.6f);
}

TEST_F(GemmaTest, CrossEntropyJingleBells) {
  float entropy = GemmaCrossEntropy(kJingleBells);
  std::cout << "per-byte entropy: " << entropy << "\n";
  EXPECT_LT(entropy, 2.3f);
}

TEST_F(GemmaTest, CrossEntropyGettysburg) {
  float entropy = GemmaCrossEntropy(kGettysburg);
  std::cout << "per-byte entropy: " << entropy << "\n";
  EXPECT_LT(entropy, 1.2f);
}

TEST_F(GemmaTest, CrossEntropyDeclaration) {
  float entropy = GemmaCrossEntropy(kDeclaration);
  std::cout << "per-byte entropy: " << entropy << "\n";
  EXPECT_LT(entropy, 1.0f);
}

}  // namespace
}  // namespace gcpp
