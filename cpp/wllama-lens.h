#pragma once

// logit lens: capture residual stream per layer, project each row through the output head

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// internal header, needed for output_norm / output weights
#include "../llama.cpp/src/llama-model.h"

struct lens_capture
{
  bool active = false;
  int64_t n_embd = 0;
  // hidden[0] is the token embedding, hidden[il + 1] is l_out-il; each is [n_tokens][n_embd]
  std::vector<std::vector<float>> hidden;

  void reset(int64_t embd, int n_layer)
  {
    n_embd = embd;
    hidden.assign(n_layer + 1, {});
  }
};

// returns slot in lens_capture::hidden, or -1
static int lens_tensor_slot(const ggml_tensor *t)
{
  // the token embedding lookup has no stable name, match the op instead
  if (t->op == GGML_OP_GET_ROWS && t->src[0] && strcmp(t->src[0]->name, "token_embd.weight") == 0)
    return 0;
  if (strncmp(t->name, "l_out-", 6) == 0)
    return atoi(t->name + 6) + 1;
  return -1;
}

static bool lens_cb_eval(ggml_tensor *t, bool ask, void *user_data)
{
  auto *cap = (lens_capture *)user_data;
  if (!cap->active)
    return !ask;
  const int slot = lens_tensor_slot(t);
  const bool want = slot >= 0 && slot < (int)cap->hidden.size() && t->type == GGML_TYPE_F32 && t->ne[0] == cap->n_embd && ggml_is_contiguous(t);
  if (ask)
    return want;
  // keep only the first embedding lookup, later ones belong to unused input paths
  if (want && !(slot == 0 && !cap->hidden[0].empty()))
  {
    auto &dst = cap->hidden[slot];
    dst.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
  }
  return true;
}

struct lens_result
{
  int n_rows = 0;
  int k = 0;
  std::vector<int32_t> top_ids;    // [n_rows][k], sorted by prob desc
  std::vector<float> top_probs;    // [n_rows][k]
  std::vector<float> entropy;      // [n_rows], nats
  std::vector<int32_t> target_ids; // [n_rows]
  std::vector<float> target_prob;  // [n_rows]
  std::vector<int32_t> target_rank; // [n_rows], 0 = top token
  std::vector<float> kl;           // [n_rows], nats, KL(final layer || row) at the same position

  void reset(int n, int k_)
  {
    n_rows = n;
    k = k_;
    top_ids.assign(n * k, 0);
    top_probs.assign(n * k, 0.0f);
    entropy.assign(n, 0.0f);
    target_ids.assign(n, 0);
    target_prob.assign(n, 0.0f);
    target_rank.assign(n, 0);
    kl.assign(n, 0.0f);
  }

  void copy_row(const lens_result &src, int s, int d)
  {
    std::copy_n(src.top_ids.begin() + s * k, k, top_ids.begin() + d * k);
    std::copy_n(src.top_probs.begin() + s * k, k, top_probs.begin() + d * k);
    entropy[d] = src.entropy[s];
    target_ids[d] = src.target_ids[s];
    target_prob[d] = src.target_prob[s];
    target_rank[d] = src.target_rank[s];
    kl[d] = src.kl[s];
  }
};

// positions per projection block; bounds the final layer log-probs kept for KL
static const int lens_pos_block = 64;

static bool lens_arch_supported(const llama_model *model, std::string &arch_out)
{
  char buf[128] = {0};
  llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
  arch_out = buf;
  static const char *allowed[] = {"llama", "qwen2", "qwen3", "qwen35"};
  for (const char *a : allowed)
    if (arch_out == a)
      return true;
  return false;
}

struct lens_projector
{
  const llama_model *model = nullptr;
  std::vector<ggml_backend_t> backends;
  ggml_backend_sched_t sched = nullptr;
  size_t max_chunk_bytes = 32u * 1024 * 1024;
  // final layer log-probs per position, kept on the output device for KL
  ggml_context *final_ctx = nullptr;
  ggml_backend_buffer_t final_buf = nullptr;
  ggml_tensor *final_logp = nullptr; // [n_vocab, n_pos]

  void init(const llama_model *m)
  {
    model = m;
    ggml_backend_dev_t dev = model->dev_output();
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU)
    {
      ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
      if (b)
        backends.push_back(b);
    }
    backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 256, false, true);
  }

  ~lens_projector()
  {
    free_final();
    if (sched)
      ggml_backend_sched_free(sched);
    for (auto b : backends)
      ggml_backend_free(b);
  }

  void free_final()
  {
    if (final_buf)
      ggml_backend_buffer_free(final_buf);
    if (final_ctx)
      ggml_free(final_ctx);
    final_buf = nullptr;
    final_ctx = nullptr;
    final_logp = nullptr;
  }

  void alloc_final(int n_pos)
  {
    if (final_logp && final_logp->ne[1] >= n_pos)
      return;
    free_final();
    ggml_init_params ip = {ggml_tensor_overhead(), nullptr, true};
    final_ctx = ggml_init(ip);
    final_logp = ggml_new_tensor_2d(final_ctx, GGML_TYPE_F32, model->output->ne[1], n_pos);
    final_buf = ggml_backend_alloc_ctx_tensors(final_ctx, backends[0]);
    if (!final_buf)
      throw std::runtime_error("lens: failed to allocate final layer buffer");
  }

  const char *backend_name() const
  {
    return backends.empty() ? "none" : ggml_backend_name(backends[0]);
  }

  // rows: [n_rows][n_embd]; targets: [n_rows] or empty (then target = top-1 of each row)
  // pos: [n_rows] row in final_logp for KL, or null; store_final: write log-probs to final_logp at row_offset
  void project_chunk(const float *rows, int n_rows, int k, const int32_t *targets, lens_result &out, int row_offset, const int32_t *pos, bool store_final)
  {
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_vocab = model->output->ne[1];

    ggml_init_params ip = {
        /*.mem_size   =*/ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context *ctx = ggml_init(ip);

    ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_rows);
    ggml_set_input(inp);
    ggml_tensor *inp_tgt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_rows);
    ggml_set_input(inp_tgt);

    ggml_tensor *x = ggml_rms_norm(ctx, inp, model->hparams.f_norm_rms_eps);
    x = ggml_mul(ctx, x, model->output_norm);
    ggml_tensor *z = ggml_mul_mat(ctx, model->output, x); // [n_vocab, n_rows]
    if (model->output_s)
      z = ggml_mul(ctx, z, model->output_s);
    ggml_tensor *p = ggml_soft_max(ctx, z);

    ggml_tensor *top = ggml_top_k(ctx, p, k); // [k, n_rows], unordered
    ggml_tensor *p3 = ggml_reshape_3d(ctx, p, 1, n_vocab, n_rows);
    ggml_tensor *top_p = ggml_get_rows(ctx, p3, top); // [1, k, n_rows]

    ggml_tensor *plogp = ggml_mul(ctx, p, ggml_log(ctx, ggml_clamp(ctx, p, 1e-30f, 1.0f)));
    ggml_tensor *ent = ggml_sum_rows(ctx, plogp); // [1, n_rows], = -entropy

    ggml_tensor *tgt_p = ggml_get_rows(ctx, p3, inp_tgt); // [1, 1, n_rows]
    ggml_tensor *z3 = ggml_reshape_3d(ctx, z, 1, n_vocab, n_rows);
    ggml_tensor *tgt_z = ggml_reshape_2d(ctx, ggml_get_rows(ctx, z3, inp_tgt), 1, n_rows);
    ggml_tensor *rank = ggml_sum_rows(ctx, ggml_step(ctx, ggml_sub(ctx, z, tgt_z))); // [1, n_rows]

    // log p = z - lse, lse = z_max - log(p_max); no clamp, so tiny p stay exact
    ggml_tensor *amax = ggml_reshape_2d(ctx, ggml_argmax(ctx, z), 1, n_rows);
    ggml_tensor *z_max = ggml_reshape_2d(ctx, ggml_get_rows(ctx, z3, amax), 1, n_rows);
    ggml_tensor *p_max = ggml_reshape_2d(ctx, ggml_get_rows(ctx, p3, amax), 1, n_rows);
    ggml_tensor *logp = ggml_sub(ctx, z, ggml_sub(ctx, z_max, ggml_log(ctx, p_max)));

    ggml_tensor *inp_pos = nullptr;
    ggml_tensor *kl = nullptr;
    if (pos)
    {
      inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
      ggml_set_input(inp_pos);
      ggml_tensor *lf = ggml_get_rows(ctx, final_logp, inp_pos); // [n_vocab, n_rows]
      kl = ggml_sum_rows(ctx, ggml_mul(ctx, ggml_exp(ctx, lf), ggml_sub(ctx, lf, logp))); // [1, n_rows]
      ggml_set_output(kl);
    }

    for (ggml_tensor *t : {top, top_p, ent, tgt_p, rank})
      ggml_set_output(t);

    ggml_cgraph *gf = ggml_new_graph(ctx);
    for (ggml_tensor *t : {top, top_p, ent, tgt_p, rank})
      ggml_build_forward_expand(gf, t);
    if (kl)
      ggml_build_forward_expand(gf, kl);
    if (store_final)
      ggml_build_forward_expand(gf, ggml_cpy(ctx, logp, ggml_view_2d(ctx, final_logp, final_logp->ne[0], n_rows, final_logp->nb[1], row_offset * final_logp->nb[1])));

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
    {
      ggml_free(ctx);
      throw std::runtime_error("lens: failed to allocate projection graph");
    }

    std::vector<int32_t> tgt(n_rows, 0);
    if (targets)
      std::copy(targets, targets + n_rows, tgt.begin());
    ggml_backend_tensor_set(inp, rows, 0, ggml_nbytes(inp));
    ggml_backend_tensor_set(inp_tgt, tgt.data(), 0, ggml_nbytes(inp_tgt));
    if (inp_pos)
      ggml_backend_tensor_set(inp_pos, pos, 0, ggml_nbytes(inp_pos));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
    {
      ggml_free(ctx);
      throw std::runtime_error("lens: projection graph compute failed");
    }

    std::vector<int32_t> h_top(k * n_rows);
    std::vector<float> h_top_p(k * n_rows), h_ent(n_rows), h_tgt_p(n_rows), h_rank(n_rows), h_kl(n_rows, 0.0f);
    ggml_backend_tensor_get(top, h_top.data(), 0, ggml_nbytes(top));
    ggml_backend_tensor_get(top_p, h_top_p.data(), 0, ggml_nbytes(top_p));
    ggml_backend_tensor_get(ent, h_ent.data(), 0, ggml_nbytes(ent));
    ggml_backend_tensor_get(tgt_p, h_tgt_p.data(), 0, ggml_nbytes(tgt_p));
    ggml_backend_tensor_get(rank, h_rank.data(), 0, ggml_nbytes(rank));
    if (kl)
      ggml_backend_tensor_get(kl, h_kl.data(), 0, ggml_nbytes(kl));
    ggml_free(ctx);

    std::vector<int> order(k);
    for (int r = 0; r < n_rows; r++)
    {
      const int dst = row_offset + r;
      for (int j = 0; j < k; j++)
        order[j] = j;
      std::sort(order.begin(), order.end(), [&](int a, int b)
                { return h_top_p[r * k + a] > h_top_p[r * k + b]; });
      for (int j = 0; j < k; j++)
      {
        out.top_ids[dst * k + j] = h_top[r * k + order[j]];
        out.top_probs[dst * k + j] = h_top_p[r * k + order[j]];
      }
      out.entropy[dst] = -h_ent[r];
      out.target_ids[dst] = tgt[r];
      out.target_prob[dst] = h_tgt_p[r];
      out.target_rank[dst] = (int32_t)h_rank[r];
      out.kl[dst] = std::max(0.0f, h_kl[r]); // rounding can give tiny negatives
    }
  }

  void project(const float *rows, int n_rows, int k, const int32_t *targets, lens_result &out, const int32_t *pos = nullptr, bool store_final = false)
  {
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_vocab = model->output->ne[1];
    const int chunk = std::max<int>(1, (int)(max_chunk_bytes / (n_vocab * sizeof(float))));

    out.reset(n_rows, k);
    if (store_final)
      alloc_final(n_rows);

    for (int r0 = 0; r0 < n_rows; r0 += chunk)
    {
      const int n = std::min(chunk, n_rows - r0);
      project_chunk(rows + r0 * n_embd, n, k, targets ? targets + r0 : nullptr, out, r0, pos ? pos + r0 : nullptr, store_final);
    }
  }
};

// layer-major result of one lens run: row = layer * n_tokens + pos, layer 0 = embeddings
struct lens_run_output
{
  int n_layers = 0; // including embedding row
  int n_tokens = 0;
  std::vector<llama_token> tokens;
  lens_result res;
  double t_decode_ms = 0;  // forward pass with capture, 0 when hidden states are reused
  double t_project_ms = 0; // output head projection and top-k
};

// rows for layers [layer_begin, layer_end) and positions [p0, p1) of the captured hidden states
static std::vector<float> lens_gather_rows(const lens_capture &cap, int n_tokens, int layer_begin, int layer_end, int p0, int p1)
{
  std::vector<float> rows;
  rows.reserve((size_t)(layer_end - layer_begin) * (p1 - p0) * cap.n_embd);
  for (int l = layer_begin; l < layer_end; l++)
  {
    const auto &h = cap.hidden[l];
    if ((int64_t)h.size() != n_tokens * cap.n_embd)
      throw std::runtime_error("lens: missing hidden state for layer slot " + std::to_string(l) + " (got " + std::to_string(h.size() / std::max<int64_t>(1, cap.n_embd)) + " rows)");
    rows.insert(rows.end(), h.begin() + p0 * cap.n_embd, h.begin() + p1 * cap.n_embd);
  }
  return rows;
}

// run the model on tokens, capture all layers, project layers [layer_begin, layer_end)
// targets: empty -> per position, use the final layer's top-1 token
static void lens_run(llama_context *lctx, lens_capture &cap, lens_projector &proj, const std::vector<llama_token> &tokens, int k, int layer_begin, int layer_end, const std::vector<int32_t> &targets_per_pos, lens_run_output &out, bool reuse_hidden = false)
{
  const llama_model *model = llama_get_model(lctx);
  const int n_layer = llama_model_n_layer(model);
  const int n_tokens = (int)tokens.size();
  if (n_tokens == 0)
    throw std::runtime_error("lens: empty prompt");
  if (n_tokens > (int)llama_n_ubatch(lctx))
    throw std::runtime_error("lens: prompt too long (" + std::to_string(n_tokens) + " > " + std::to_string(llama_n_ubatch(lctx)) + " tokens)");
  layer_begin = std::max(0, layer_begin);
  layer_end = (layer_end <= 0 || layer_end > n_layer + 1) ? n_layer + 1 : layer_end;
  if (layer_begin >= layer_end)
    throw std::runtime_error("lens: empty layer range");

  if (!reuse_hidden)
  {
    cap.reset(llama_model_n_embd(model), n_layer);
    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++)
    {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = true; // keep all rows through the last layer
    }
    batch.n_tokens = n_tokens;
    cap.active = true;
    const int64_t t0 = ggml_time_us();
    const int ret = llama_decode(lctx, batch);
    out.t_decode_ms = (ggml_time_us() - t0) / 1000.0;
    cap.active = false;
    llama_batch_free(batch);
    if (ret != 0)
      throw std::runtime_error("lens: llama_decode failed with code " + std::to_string(ret));
  }

  const int64_t t_proj0 = ggml_time_us();
  if (!targets_per_pos.empty() && (int)targets_per_pos.size() != n_tokens)
    throw std::runtime_error("lens: targets must have one entry per token");
  // reuse with given targets: hidden states and KL are unchanged, so skip the final layer pass
  const bool with_final = !reuse_hidden || targets_per_pos.empty();
  const int n_layers = layer_end - layer_begin;
  out.n_layers = n_layers;
  out.n_tokens = n_tokens;
  out.tokens = tokens;
  out.res.reset(n_layers * n_tokens, k);
  for (int p0 = 0; p0 < n_tokens; p0 += lens_pos_block)
  {
    const int np = std::min(lens_pos_block, n_tokens - p0);
    std::vector<int32_t> targets;
    if (!targets_per_pos.empty())
      targets.assign(targets_per_pos.begin() + p0, targets_per_pos.begin() + p0 + np);
    if (with_final)
    {
      // final layer first: keeps its log-probs for KL and gives the default targets
      std::vector<float> last = lens_gather_rows(cap, n_tokens, n_layer, n_layer + 1, p0, p0 + np);
      lens_result r;
      proj.project(last.data(), np, 1, nullptr, r, nullptr, true);
      if (targets.empty())
        targets = r.top_ids; // final layer top-1 per position
    }

    std::vector<float> rows = lens_gather_rows(cap, n_tokens, layer_begin, layer_end, p0, p0 + np);
    std::vector<int32_t> row_targets, row_pos;
    row_targets.reserve(n_layers * np);
    row_pos.reserve(n_layers * np);
    for (int l = 0; l < n_layers; l++)
    {
      row_targets.insert(row_targets.end(), targets.begin(), targets.end());
      for (int t = 0; t < np; t++)
        row_pos.push_back(t);
    }
    lens_result r;
    proj.project(rows.data(), n_layers * np, k, row_targets.data(), r, with_final ? row_pos.data() : nullptr);
    for (int l = 0; l < n_layers; l++)
      for (int t = 0; t < np; t++)
        out.res.copy_row(r, l * np + t, l * n_tokens + p0 + t);
  }
  out.t_project_ms = (ggml_time_us() - t_proj0) / 1000.0;
}
