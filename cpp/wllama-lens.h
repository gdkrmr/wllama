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
// staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith, for the MTP draft
#include "../llama.cpp/src/llama-ext.h"

// tokens per llama_decode call; bounds the logits buffers llama.cpp allocates (n_vocab floats per token)
static const int lens_batch = 64;

// MTP (NextN) block states captured in the MTP context: after eh_proj, after attention, block output
static const char *lens_mtp_names[] = {"mtp_eh_proj-", "mtp_attn_residual-", "mtp_post_ffn-"};
static const int lens_mtp_stages = 3;

struct lens_capture
{
  bool active = false;
  bool got_embd = false; // per decode call
  bool stop = false;     // per decode call, set after the last layer
  bool full = false;     // do not stop after the last slot, run the whole graph
  bool mtp = false;      // MTP context: hidden[s] is stage s of lens_mtp_names
  bool with_h = false;   // main context: also keep h_nextn, the MTP input (final norm of the last layer)
  int64_t n_embd = 0;
  // hidden[0] is the token embedding, hidden[il + 1] is l_out-il; each is [n_tokens][n_embd]
  std::vector<std::vector<float>> hidden;
  std::vector<float> h_nextn; // [n_tokens][n_embd] when with_h

  void reset(int64_t embd, int n_layer)
  {
    n_embd = embd;
    hidden.assign(n_layer + 1, {});
  }
};

// returns slot in lens_capture::hidden, hidden.size() for h_nextn, or -1
static int lens_tensor_slot(const ggml_tensor *t, const lens_capture *cap)
{
  if (cap->mtp)
  {
    for (int s = 0; s < lens_mtp_stages; s++)
      if (strncmp(t->name, lens_mtp_names[s], strlen(lens_mtp_names[s])) == 0)
        return s;
    return -1;
  }
  // the token embedding lookup has no stable name, match the op instead
  if (t->op == GGML_OP_GET_ROWS && t->src[0] && strcmp(t->src[0]->name, "token_embd.weight") == 0)
    return 0;
  if (strncmp(t->name, "l_out-", 6) == 0)
    return atoi(t->name + 6) + 1;
  // h_nextn gets renamed to result_norm when it is not cropped
  if (cap->with_h && (strcmp(t->name, "h_nextn") == 0 || strcmp(t->name, "result_norm") == 0))
    return (int)cap->hidden.size();
  return -1;
}

static bool lens_cb_eval(ggml_tensor *t, bool ask, void *user_data)
{
  auto *cap = (lens_capture *)user_data;
  if (!cap->active)
    return !ask;
  // after the last layer: run only the first node of each remaining split, then break it
  if (cap->stop)
    return ask;
  const int slot = lens_tensor_slot(t, cap);
  const int last = (int)cap->hidden.size() - (cap->with_h ? 0 : 1);
  const bool want = slot >= 0 && slot <= last && t->type == GGML_TYPE_F32 && t->ne[0] == cap->n_embd && ggml_is_contiguous(t);
  if (ask)
    return want;
  // keep only the first embedding lookup, later ones belong to unused input paths
  if (want && !(slot == 0 && !cap->mtp && cap->got_embd))
  {
    auto &dst = slot < (int)cap->hidden.size() ? cap->hidden[slot] : cap->h_nextn;
    const size_t n0 = dst.size();
    dst.resize(n0 + ggml_nelements(t));
    ggml_backend_tensor_get(t, dst.data() + n0, 0, ggml_nbytes(t));
    cap->got_embd |= slot == 0;
    // false stops the rest of this split, so the output head is not computed
    cap->stop = !cap->full && slot == last;
  }
  return !cap->stop;
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
  static const char *allowed[] = {"llama", "qwen2", "qwen3", "qwen35", "gemma", "gemma2", "gemma3", "gemma4", "phi2", "phi3"};
  for (const char *a : allowed)
    if (arch_out == a)
      return true;
  return false;
}

// MTP needs the NextN block loaded (llama_model_params::load_mtp); only the qwen35 MTP graph is known to work
static bool lens_has_mtp(const llama_model *model)
{
  return model->arch == LLM_ARCH_QWEN35 && model->hparams.n_layer_nextn > 0 && model->layers[model->hparams.n_layer()].nextn.eh_proj;
}

static bool lens_is_gemma(const llama_model *model)
{
  return model->arch == LLM_ARCH_GEMMA || model->arch == LLM_ARCH_GEMMA2 || model->arch == LLM_ARCH_GEMMA3 || model->arch == LLM_ARCH_GEMMA4;
}

// final norm and output matrix that turn a hidden state into logits
struct lens_head
{
  ggml_tensor *norm = nullptr, *norm_b = nullptr, *out = nullptr, *out_s = nullptr, *out_b = nullptr;
};

struct lens_projector
{
  const llama_model *model = nullptr;
  lens_head main_head, mtp_head; // mtp_head.out is null without an MTP block
  std::vector<ggml_backend_t> backends;
  ggml_backend_sched_t sched = nullptr;
  size_t max_chunk_bytes = 32u * 1024 * 1024;
  // final layer log-probs per position, kept on the output device for KL
  ggml_context *final_ctx = nullptr;
  ggml_backend_buffer_t final_buf = nullptr;
  ggml_tensor *final_logp = nullptr; // [n_vocab, n_pos]

  // n_threads: CPU backend threads, 0 = ggml default; in wasm it must not exceed the pthread pool
  void init(const llama_model *m, int n_threads = 0)
  {
    model = m;
    main_head = {model->output_norm, model->output_norm_b, model->output, model->output_s, model->output_b};
    if (lens_has_mtp(model))
    {
      // same choice as the qwen35 MTP graph: shared head if present, else the main one
      const auto &nextn = model->layers[model->hparams.n_layer()].nextn;
      mtp_head = {nextn.shared_head_norm ? nextn.shared_head_norm : model->output_norm, nullptr,
                  nextn.shared_head_head ? nextn.shared_head_head : model->output,
                  nextn.shared_head_head ? nextn.shared_head_head_s : model->output_s, nullptr};
    }
    ggml_backend_dev_t dev = model->dev_output();
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU)
    {
      ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
      if (b)
        backends.push_back(b);
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (n_threads > 0)
      ggml_backend_cpu_set_n_threads(cpu, n_threads);
    backends.push_back(cpu);
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
  void project_chunk(const float *rows, int n_rows, int k, const int32_t *targets, lens_result &out, int row_offset, const int32_t *pos, bool store_final, const lens_head &hd)
  {
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_vocab = hd.out->ne[1];

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

    // phi2 uses LayerNorm, the others RMSNorm
    ggml_tensor *x = model->arch == LLM_ARCH_PHI2 ? ggml_norm(ctx, inp, model->hparams.f_norm_eps) : ggml_rms_norm(ctx, inp, model->hparams.f_norm_rms_eps);
    x = ggml_mul(ctx, x, hd.norm);
    if (hd.norm_b)
      x = ggml_add(ctx, x, hd.norm_b);
    ggml_tensor *z = ggml_mul_mat(ctx, hd.out, x); // [n_vocab, n_rows]
    if (hd.out_s)
      z = ggml_mul(ctx, z, hd.out_s);
    if (hd.out_b)
      z = ggml_add(ctx, z, hd.out_b);
    // gemma 1 has no soft-capping, its hparam keeps a non-zero default
    const float softcap = lens_is_gemma(model) && model->arch != LLM_ARCH_GEMMA ? model->hparams.f_final_logit_softcapping : 0.0f;
    if (softcap > 0.0f)
      z = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, z, 1.0f / softcap)), softcap);
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

  // mtp: use the MTP block's head; no KL then
  void project(const float *rows, int n_rows, int k, const int32_t *targets, lens_result &out, const int32_t *pos = nullptr, bool store_final = false, bool mtp = false)
  {
    const lens_head &hd = mtp ? mtp_head : main_head;
    if (!hd.out)
      throw std::runtime_error("lens: model has no MTP head");
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_vocab = hd.out->ne[1];
    const int chunk = std::max<int>(1, (int)(max_chunk_bytes / (n_vocab * sizeof(float))));

    out.reset(n_rows, k);
    if (store_final)
      alloc_final(n_rows);

    for (int r0 = 0; r0 < n_rows; r0 += chunk)
    {
      const int n = std::min(chunk, n_rows - r0);
      project_chunk(rows + r0 * n_embd, n, k, targets ? targets + r0 : nullptr, out, r0, pos ? pos + r0 : nullptr, store_final, hd);
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
  // MTP: row = stage * n_tokens + t, stage as in lens_mtp_names; the NextN block reads h at t and token t + 1 and predicts token t + 2
  // the last position has no token t + 1, its top ids are -1
  int n_mtp = 0; // 0 = no MTP rows
  lens_result mtp;
  // chained MTP draft after the prompt: step j reads draft_tokens[j] (x1 = main model top-1 at the last position) and guesses draft_tokens[j + 1]
  // row = stage * n_draft + j
  int n_draft = 0;
  std::vector<llama_token> draft_tokens; // n_draft + 1 entries, or empty
  lens_result draft;
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

// decode tokens [i0, i0 + n) at their positions; all rows are outputs, else the last layer is cropped to the output rows
static int lens_decode(llama_context *lctx, const std::vector<llama_token> &tokens, int i0, int n)
{
  llama_batch batch = llama_batch_init(n, 0, 1);
  for (int i = 0; i < n; i++)
  {
    batch.token[i] = tokens[i0 + i];
    batch.pos[i] = i0 + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = true;
  }
  batch.n_tokens = n;
  const int ret = llama_decode(lctx, batch);
  llama_batch_free(batch);
  return ret;
}

// MTP context (ctx_type LLAMA_CONTEXT_TYPE_MTP, cb_eval = lens_cb_eval with cap); it holds its own KV cache of the NextN block
struct lens_mtp
{
  llama_context *ctx = nullptr;
  lens_capture cap; // cap.mtp = true; row p is the MTP input (h at p - 1, token p)
  int n_draft = 0;   // tokens to draft after the prompt
};

// MTP rows [i0, i0 + n): token p with h_nextn of p - 1; row 0 gets a zero h, like llama.cpp's MTP draft
static int lens_mtp_decode(llama_context *mctx, const std::vector<llama_token> &tokens, const std::vector<float> &h, int64_t n_embd, int i0, int n, bool all_outputs)
{
  const std::vector<float> zeros(n_embd, 0.0f);
  llama_batch_ext *batch = llama_batch_ext_init(mctx);
  for (int i = 0; i < n; i++)
  {
    const llama_pos p = i0 + i;
    const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[p]);
    llama_batch_ext_set_pos(batch, idx, &p);
    llama_batch_ext_set_embd_state(batch, idx, {p == 0 ? zeros.data() : h.data() + (p - 1) * n_embd, 1, (size_t)n_embd});
    llama_batch_ext_set_output_logits(batch, idx, all_outputs || i == n - 1);
  }
  const int ret = llama_process(mctx, LLAMA_PROCESS_TYPE_DECODE, batch);
  llama_batch_ext_free(batch);
  return ret;
}

// chained MTP draft like llama.cpp's draft-mtp (common/speculative.cpp): step j decodes draft token j at pos0 + j with the h_nextn of the step before, the guess is the argmax
// h: h_nextn of the main model at pos0 - 1; the draft rows are removed from the MTP KV cache again
static void lens_mtp_draft(lens_mtp &mtp, lens_projector &proj, const float *h, llama_token x1, llama_pos pos0, int k, lens_run_output &out)
{
  llama_context *mctx = mtp.ctx;
  const int64_t n_embd = mtp.cap.n_embd;
  const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(mctx)));
  std::vector<float> h_cur(h, h + n_embd);
  std::vector<std::vector<float>> forced = std::move(mtp.cap.hidden);
  mtp.cap.hidden.assign(lens_mtp_stages, {});
  out.draft_tokens = {x1};
  llama_set_embeddings_nextn(mctx, true, true);
  int ret = 0;
  for (int j = 0; j < mtp.n_draft && pos0 + j < (llama_pos)llama_n_ctx(mctx); j++)
  {
    const llama_pos p = pos0 + j;
    llama_batch_ext *batch = llama_batch_ext_init(mctx);
    const int32_t idx = llama_batch_ext_add_token(batch, 0, out.draft_tokens.back());
    llama_batch_ext_set_pos(batch, idx, &p);
    llama_batch_ext_set_embd_state(batch, idx, {h_cur.data(), 1, (size_t)n_embd});
    llama_batch_ext_set_output_logits(batch, idx, true);
    mtp.cap.active = mtp.cap.full = true;
    mtp.cap.stop = false;
    ret = llama_process(mctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    mtp.cap.active = mtp.cap.full = false;
    llama_batch_ext_free(batch);
    if (ret != 0)
      break;
    const float *logits = llama_get_logits_ith(mctx, idx);
    llama_token best = 0;
    for (int v = 1; v < n_vocab; v++)
      if (logits[v] > logits[best])
        best = v;
    out.draft_tokens.push_back(best);
    const float *hn = llama_get_embeddings_nextn_ith(mctx, idx);
    h_cur.assign(hn, hn + n_embd);
  }
  llama_set_embeddings_nextn(mctx, false, false);
  llama_memory_seq_rm(llama_get_memory(mctx), 0, pos0, -1);
  std::vector<std::vector<float>> rows_cap = std::move(mtp.cap.hidden);
  mtp.cap.hidden = std::move(forced);
  if (ret != 0)
    throw std::runtime_error("lens: MTP draft decode failed with code " + std::to_string(ret));
  out.n_draft = (int)out.draft_tokens.size() - 1;
  if (out.n_draft == 0)
  {
    out.draft_tokens.clear();
    return;
  }
  std::vector<float> rows;
  for (const auto &r : rows_cap)
    rows.insert(rows.end(), r.begin(), r.end());
  if ((int64_t)rows.size() != lens_mtp_stages * out.n_draft * n_embd)
    throw std::runtime_error("lens: missing MTP draft states");
  proj.project(rows.data(), lens_mtp_stages * out.n_draft, k, nullptr, out.draft, nullptr, false, true);
}

// run the model on tokens, capture all layers, project layers [layer_begin, layer_end)
// targets: empty -> per position, use the final layer's top-1 token
// incremental: out holds the last result of this context; keep the positions that tokens still share, run only the rest
// mtp: also run the NextN block and fill out.mtp; null = no MTP
static void lens_run(llama_context *lctx, lens_capture &cap, lens_projector &proj, const std::vector<llama_token> &tokens, int k, int layer_begin, int layer_end, const std::vector<int32_t> &targets_per_pos, lens_run_output &out, bool reuse_hidden = false, bool incremental = false, lens_mtp *mtp = nullptr)
{
  const llama_model *model = llama_get_model(lctx);
  const int n_layer = llama_model_n_layer(model);
  const int n_tokens = (int)tokens.size();
  if (n_tokens == 0)
    throw std::runtime_error("lens: empty prompt");
  if (n_tokens > (int)llama_n_ctx(lctx))
    throw std::runtime_error("lens: prompt too long (" + std::to_string(n_tokens) + " > " + std::to_string(llama_n_ctx(lctx)) + " tokens)");
  layer_begin = std::max(0, layer_begin);
  layer_end = (layer_end <= 0 || layer_end > n_layer + 1) ? n_layer + 1 : layer_end;
  if (layer_begin >= layer_end)
    throw std::runtime_error("lens: empty layer range");

  llama_memory_t mem = llama_get_memory(lctx);
  const int n_old = out.n_tokens;
  int n_keep = 0;
  if (incremental && !reuse_hidden && targets_per_pos.empty() && layer_begin == 0 && layer_end == n_layer + 1 && out.n_layers == n_layer + 1 && out.res.k == k)
  {
    while (n_keep < n_old && n_keep < n_tokens && out.tokens[n_keep] == tokens[n_keep])
      n_keep++;
    // memory and hidden states must hold exactly the old tokens
    bool ok = n_keep > 0 && n_keep < n_tokens && llama_memory_seq_pos_max(mem, 0) == n_old - 1 && (int)cap.hidden.size() == n_layer + 1;
    for (const auto &h : cap.hidden)
      ok = ok && (int64_t)h.size() == n_old * cap.n_embd;
    if (mtp)
    {
      ok = ok && cap.with_h && (int64_t)cap.h_nextn.size() == n_old * cap.n_embd && llama_memory_seq_pos_max(llama_get_memory(mtp->ctx), 0) == n_old - 1;
      for (const auto &h : mtp->cap.hidden)
        ok = ok && (int64_t)h.size() == n_old * cap.n_embd;
      ok = ok && llama_memory_seq_rm(llama_get_memory(mtp->ctx), 0, n_keep, -1);
    }
    // recurrent memory cannot drop the tail, then seq_rm fails and we run all tokens
    if (!ok || !llama_memory_seq_rm(mem, 0, n_keep, -1))
      n_keep = 0;
  }

  if (!reuse_hidden)
  {
    if (n_keep > 0)
    {
      for (auto &h : cap.hidden)
        h.resize(n_keep * cap.n_embd);
      cap.h_nextn.resize(mtp ? n_keep * cap.n_embd : 0);
      if (mtp)
        for (auto &h : mtp->cap.hidden)
          h.resize(n_keep * cap.n_embd);
    }
    else
    {
      cap.reset(llama_model_n_embd(model), n_layer);
      cap.h_nextn.clear();
      llama_memory_clear(mem, true);
      if (mtp)
      {
        mtp->cap.reset(cap.n_embd, lens_mtp_stages - 1);
        mtp->cap.mtp = true;
        llama_memory_clear(llama_get_memory(mtp->ctx), true);
      }
    }
    cap.with_h = mtp != nullptr;
    const int n_batch = std::min<int>(llama_n_batch(lctx), llama_n_ubatch(lctx));
    const int64_t t0 = ggml_time_us();
    for (int i0 = n_keep; i0 < n_tokens; i0 += n_batch)
    {
      cap.active = true;
      cap.got_embd = false;
      cap.stop = false;
      const int ret = lens_decode(lctx, tokens, i0, std::min(n_batch, n_tokens - i0));
      cap.active = false;
      if (ret != 0)
        throw std::runtime_error("lens: llama_decode failed with code " + std::to_string(ret));
    }
    for (int i0 = n_keep; mtp && i0 < n_tokens; i0 += n_batch)
    {
      mtp->cap.active = true;
      mtp->cap.stop = false;
      const int ret = lens_mtp_decode(mtp->ctx, tokens, cap.h_nextn, cap.n_embd, i0, std::min(n_batch, n_tokens - i0), false);
      mtp->cap.active = false;
      if (ret != 0)
        throw std::runtime_error("lens: MTP decode failed with code " + std::to_string(ret));
    }
    out.t_decode_ms = (ggml_time_us() - t0) / 1000.0;
    // gemma scales the embeddings after the lookup we capture; kept rows are already scaled
    if (lens_is_gemma(model))
      for (auto it = cap.hidden[0].begin() + (size_t)n_keep * cap.n_embd; it != cap.hidden[0].end(); ++it)
        *it *= sqrtf((float)cap.n_embd);
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
  lens_result prev = std::move(out.res);
  lens_result prev_mtp = std::move(out.mtp);
  out.n_mtp = 0;
  out.n_draft = 0;
  out.draft_tokens.clear();
  llama_token x1 = -1; // final layer top-1 at the last position
  out.res.reset(n_layers * n_tokens, k);
  for (int l = 0; l < n_layers; l++)
    for (int t = 0; t < n_keep; t++)
      out.res.copy_row(prev, l * n_old + t, l * n_tokens + t);
  for (int p0 = n_keep; p0 < n_tokens; p0 += lens_pos_block)
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
      if (p0 + np == n_tokens)
        x1 = r.top_ids[np - 1];
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

  // MTP column t comes from MTP row t + 1; reused hidden states may predate MTP
  bool mtp_ok = mtp != nullptr && (int)mtp->cap.hidden.size() == lens_mtp_stages;
  for (int s = 0; mtp_ok && s < lens_mtp_stages; s++)
    mtp_ok = (int64_t)mtp->cap.hidden[s].size() == n_tokens * cap.n_embd;
  if (mtp_ok)
  {
    out.n_mtp = lens_mtp_stages;
    out.mtp.reset(lens_mtp_stages * n_tokens, k);
    const int t_keep = prev_mtp.n_rows == lens_mtp_stages * n_old ? std::max(0, n_keep - 1) : 0;
    for (int s = 0; s < lens_mtp_stages; s++)
    {
      for (int t = 0; t < t_keep; t++)
        out.mtp.copy_row(prev_mtp, s * n_old + t, s * n_tokens + t);
      std::fill_n(out.mtp.top_ids.begin() + (s * n_tokens + n_tokens - 1) * k, k, -1);
    }
    for (int p0 = t_keep + 1; p0 < n_tokens; p0 += lens_pos_block)
    {
      const int np = std::min(lens_pos_block, n_tokens - p0);
      std::vector<float> rows = lens_gather_rows(mtp->cap, n_tokens, 0, lens_mtp_stages, p0, p0 + np);
      lens_result r;
      proj.project(rows.data(), lens_mtp_stages * np, k, nullptr, r, nullptr, false, true);
      for (int s = 0; s < lens_mtp_stages; s++)
        for (int i = 0; i < np; i++)
          out.mtp.copy_row(r, s * np + i, s * n_tokens + p0 + i - 1);
    }
  }
  int64_t t_draft = 0;
  if (mtp_ok && !reuse_hidden && mtp->n_draft > 0 && x1 >= 0)
  {
    t_draft = ggml_time_us();
    lens_mtp_draft(*mtp, proj, cap.h_nextn.data() + (n_tokens - 1) * cap.n_embd, x1, n_tokens, k, out);
    t_draft = ggml_time_us() - t_draft;
    out.t_decode_ms += t_draft / 1000.0;
  }
  out.t_project_ms = (ggml_time_us() - t_proj0 - t_draft) / 1000.0;
}
