#include "gemma/configs.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"  // Type
#include "io/fields.h"           // Type

namespace gcpp {

TEST(ConfigsTest, TestAll) {
  ForEachModel([&](Model model) {
    ModelConfig config(model, Type::kSFP, ChooseWrapping(model));
    fprintf(stderr, "Testing %s (%s)\n", config.display_name.c_str(),
            config.Specifier().c_str());
    HWY_ASSERT(config.model == model);

    // We can deduce the model/display_name from all other fields.
    config.model = Model::UNKNOWN;
    const std::string saved_display_name = config.display_name;
    config.display_name.clear();
    HWY_ASSERT(config.OverwriteWithCanonical());
    HWY_ASSERT(config.model == model);
    HWY_ASSERT(config.display_name == saved_display_name);

    const std::vector<uint32_t> serialized = config.Write();
    ModelConfig deserialized;
    const IFields::ReadResult result =
        deserialized.Read(hwy::Span<const uint32_t>(serialized), /*pos=*/0);
    HWY_ASSERT(result.pos == serialized.size());
    // We wrote it, so all fields should be known, and no extra.
    HWY_ASSERT(result.extra_u32 == 0);
    HWY_ASSERT(result.missing_fields == 0);
    // All fields should match.
    HWY_ASSERT(deserialized.TestEqual(config, /*print=*/true));
    HWY_ASSERT(deserialized.model == model);
    HWY_ASSERT(deserialized.display_name == saved_display_name);
  });
}

}  // namespace gcpp
