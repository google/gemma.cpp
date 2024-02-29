#pragma once

/**
 * This is a simple alternative cross-platform implementation of getopt, which
 * is used to parse argument strings submitted to the executable (argc and argv
 * which are received in the main function).
 */

#ifndef CAG_LIBRARY_H
#define CAG_LIBRARY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * An option is used to describe a flag/argument option submitted when the
 * program is run.
 */
typedef struct cag_option
{
  const char identifier;
  const char *access_letters;
  const char *access_name;
  const char *value_name;
  const char *description;
} cag_option;

/**
 * A context is used to iterate over all options provided. It stores the parsing
 * state.
 */
typedef struct cag_option_context
{
  const struct cag_option *options;
  size_t option_count;
  int argc;
  char **argv;
  int index;
  int inner_index;
  bool forced_end;
  char identifier;
  char *value;
} cag_option_context;

/**
 * This is just a small macro which calculates the size of an array.
 */
#define CAG_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))


/**
 * @brief Prints all options to the terminal.
 *
 * This function prints all options to the terminal. This can be used to
 * generate the output for a "--help" option.
 *
 * @param options The options which will be printed.
 * @param option_count The option count which will be printed.
 * @param destination The destination where the output will be printed.
 */
void cag_option_print(const cag_option *options, size_t option_count,
  FILE *destination);

/**
 * @brief Prepare argument options context for parsing.
 *
 * This function prepares the context for iteration and initializes the context
 * with the supplied options and arguments. After the context has been prepared,
 * it can be used to fetch arguments from it.
 *
 * @param context The context which will be initialized.
 * @param options The registered options which are available for the program.
 * @param option_count The amount of options which are available for the
 * program.
 * @param argc The amount of arguments the user supplied in the main function.
 * @param argv A pointer to the arguments of the main function.
 */
void cag_option_prepare(cag_option_context *context, const cag_option *options,
  size_t option_count, int argc, char **argv);

/**
 * @brief Fetches an option from the argument list.
 *
 * This function fetches a single option from the argument list. The context
 * will be moved to that item. Information can be extracted from the context
 * after the item has been fetched.
 * The arguments will be re-ordered, which means that non-option arguments will
 * be moved to the end of the argument list. After all options have been
 * fetched, all non-option arguments will be positioned after the index of
 * the context.
 *
 * @param context The context from which we will fetch the option.
 * @return Returns true if there was another option or false if the end is
 * reached.
 */
bool cag_option_fetch(cag_option_context *context);

/**
 * @brief Gets the identifier of the option.
 *
 * This function gets the identifier of the option, which should be unique to
 * this option and can be used to determine what kind of option this is.
 *
 * @param context The context from which the option was fetched.
 * @return Returns the identifier of the option.
 */
char cag_option_get(const cag_option_context *context);

/**
 * @brief Gets the value from the option.
 *
 * This function gets the value from the option, if any. If the option does not
 * contain a value, this function will return NULL.
 *
 * @param context The context from which the option was fetched.
 * @return Returns a pointer to the value or NULL if there is no value.
 */
const char *cag_option_get_value(const cag_option_context *context);

/**
 * @brief Gets the current index of the context.
 *
 * This function gets the index within the argv arguments of the context. The
 * context always points to the next item which it will inspect. This is
 * particularly useful to inspect the original argument array, or to get
 * non-option arguments after option fetching has finished.
 *
 * @param context The context from which the option was fetched.
 * @return Returns the current index of the context.
 */
int cag_option_get_index(const cag_option_context *context);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
