library(ellmer)
library(vitals)
library(tidyverse)

OPENAI_MODELS <- 
  c(
    "gpt_o1" = "o1-2024-12-17",
    "gpt_o3_mini" = "o3-mini-2025-01-31",
    "gpt_o3" = "o3-2025-04-16",
    "gpt_o4_mini" = "o4-mini-2025-04-16",
    "gpt_4_1" = "gpt-4.1-2025-04-14"
  )


vitals::vitals_log_dir_set("./logs")

sonnet_3_7 <- chat_anthropic(model = "claude-3-7-sonnet-latest")

are_task <- Task$new(
  dataset = are,
  solver = generate(),
  scorer = model_graded_qa(
    scorer_chat = sonnet_3_7, 
    partial_credit = TRUE
  ),
  epochs = 3,
  name = "An R Eval"
)
  

model_eval <- function(model, filename = model, chat_fun, overwrite = FALSE, ...) {
  model_path <- fs::path("results", filename, ext = "rda")
  
  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: file already exists at {model_path}"))
    return(invisible(NULL))
  }

  chat <- chat_fun(model = model, ...)

  are_model <- are_task$clone()
  are_model$eval(solver_chat = chat)

  save(are_model, file = model_path)
}

iwalk(OPENAI_MODELS, model_eval, chat_fun = chat_openai)

model_eval("claude-3-7-sonnet-latest", filename = "sonnet_3_7", chat_fun = chat_anthropic)
model_eval(
  "claude-3-7-sonnet-latest", 
  filename = "sonnet_3_7_thinking",
  chat_fun = chat_anthropic, 
  api_args = list(
    thinking = list(type = "enabled", budget_tokens = 2000)
  )
)

