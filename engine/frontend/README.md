# 文本前端和请求表（2026-11-08）

动机：阶段 2 第 2 步。把前端三个单元和调度宿主里的请求表接到 #19 的契约和 #23 的数据通道上，调度器（第 3 步）可以照着这里的接口接进来。代码从已关闭的 #22 搬来，收发改走宿主的数据通道，出错处理改成不会永久失效。验证用 CPU、字节 tokenizer 和测试夹具里的脚本调度器，走真实的 HTTP、SSE、单元运行时和 TCP 记录流；没有模型，也没有 GPU。

## 搬运来源

- mini-sglang `9a91cfa`，MIT，许可证见 `LICENSE.mini-sglang`：
  - `python/minisgl/server/api_server.py` → `frontend/http/api.py`：FastAPI、FrontendManager 的 ack/event 表、listener、等待响应、StreamingResponse 和取消流程。
  - `python/minisgl/tokenizer/tokenize.py` → `frontend/tokenizer/unit.py`：HF chat template → encode。
  - `python/minisgl/tokenizer/detokenize.py` → `frontend/detok/unit.py`：read/surrounding offset 的增量反分词，以及不完整 UTF-8 的处理。
- SGLang `e838360`，Apache-2.0，许可证见 `engine/sched/reqs/LICENSE.sglang`：
  - `python/sglang/srt/managers/schedule_batch.py` 的请求结束逻辑 → `sched/reqs/unit.py`：`output_ids`、`finished_reason`、`finished_len`；同一步里 stop token 优先于长度上限，并裁掉多接受的 token。
- 适配：
  - 去掉上游模块级的全局 app 和 manager，状态都放在单元实例里；用 `ctx.effect` 管服务线程。
  - 上游的消息换成 `Requests` / `TokenEvents`。
  - 加了系统身份、取消后忽略迟到的结果、按 epoch 延迟回收。
  - 上游最小前端里 stop 字符串没实现、usage 是 0、finish_reason 固定是 stop，这里补成了能用的行为。

## 单元

- `frontend.tokenizer`：`frontend.tokenizer@1`，只加载本地 tokenizer，不加载权重。用 `uv sync --extra frontend` 安装依赖，设置 `frontend.tokenizer_path`。
- `frontend.detok`：`frontend.detok@1`，按请求做增量文本，处理跨分片的 stop 字符串。
- `frontend.http`：HTTP/SSE 服务，`frontend.http@1` 给出实际监听的端口。uvicorn/asyncio 跑在一个标准库线程上，宿主的控制循环照常运行。
- `sched.reqs`：`sched.reqs@1`，调度宿主里的请求事实表。流程是 `add → start_step → sample → retire → reap`：
  - `cancel` 立即结束对客户端的输出，但 `reap` 必须等这个请求提交过的步都退休；
  - 请求表不分配也不释放 GPU 资源。

支持 `/v1/completions`、`/v1/chat/completions` 的文本请求，非流式和 SSE 两种响应，以及 `/v1/models`、`/health`、`DELETE /v1/requests/{x-request-id}`：
- 只支持 n=1，非零的 presence/frequency penalty 明确拒绝。
- 每个响应带 `x-engine-identity` 和 `x-request-id`；最终结果带 `engine_identities` 和 `engine_finish_reason`。
- 客户端断开、stop 字符串命中、显式取消，都通过 `AbortRequest` 通知调度器。
- 调度器拒绝的请求（`TokenEvents.errors`）：非流式回 400，SSE 发一个错误事件。

## 两条记录流

HTTP 跑在自己的线程上，而 ZMQ socket 不能被两个线程同时用，所以前端的收发都在 `StepChannel` 自己的一个 I/O 线程上做，HTTP 协程不碰 socket（`frontend/transport.py`）。

这个线程跑一个循环，每一轮做三件事：从队列里取待发的请求，按顺序、不等待地写出去；读 token，交给 asyncio 那边；处理关闭。
- 一个请求暂时发不出去（没有调度器、连接还在建立、对方队列满了），就留在队列里，下一轮再试，读 token 照常进行。
- 每个请求的截止时间（`frontend.io_timeout_s`）从它进队列那一刻算起，排在别的请求后面等的时间也算在内；过了截止时间，这个请求回 503。
- 查进程管家最多等 0.25 秒，不会长时间卡住读 token。

两条流：

- **token 读端**：宿主的数据通道 `proc.streams@1` 里名叫 `tokens` 的读端。单元加载时在宿主线程上打开，之后只交给 I/O 线程用（[ipc.md](../../doc/specs/ipc.md) §6）。它不随前端单元重载关掉，所以调度器写入的返回地址（`ReturnRoute`，每条请求记录都带着）在重载后仍然有效。
- **请求写端**：I/O 线程自己的 `ChannelWriter`，用它自己的 ZMQ context 和自己连进程管家的 RPC 客户端，查找时不碰宿主线程的 socket。
  - 调度器的位置有两种给法：`frontend.scheduler`（ProcId，经进程管家查它的读端 `requests`，进程被换掉后会重新查），或者 `frontend.requests_address`（固定地址）。
  - 读端看到新的 token 写端、进程管家又给出了新地址时，丢掉还连着旧地址的写端，下一次直接连新地址。

**调度器被替换**：新的调度器进程不知道旧进程手里的请求。
- 一个调度器实例靠它的读端 `requests` 的地址来认：新进程一定是新地址。
- 请求发到了新地址；或者 token 读端出现了新的写端，并且进程管家给出了新地址。这两种情况下，`incarnation` 加 1。
- 前端结束发给旧实例的请求：回 503，消息是"the scheduler was replaced"。还没发出去的请求不算旧实例的，发出去以后再单独检查一次。之后的请求照常发给新实例。
- 只出现新的写端、地址没变，说明还是同一个调度器重新写了一次，不算替换。

请求表的用法：
- `start_step` 固定这一步的 row → request 映射；
- `sample` 接受采样结果，但不释放资源，调度器收齐所有参与 rank 的结果后才调 `retire`；
- `retire` 返回已经结束、又没有在途步的请求，调度器处理完自己的账本，再 `reap`；
- 没有在途步的排队请求，取消后可以直接 `reap`。

## 和 #22 的不同

- 前端不再自己建流、自己起 RPC 客户端去查 token 读端：读端用宿主的，写端跟着调度器的地址走。
- 去掉了 `frontend.tokens_bind` 和 `frontend.tokens_shm` 两个配置项，绑定地址和共享内存名都由宿主管。
- 请求登记以后、发出去之前出了任何问题（发送出错、客户端断开），只有这个请求失败，并撤掉登记，不占 `frontend.max_requests` 的名额。`seed`、`top_k` 超出能编码的范围时回 422。
- 连不上调度器不再是永久失败：以前一出错，之后一律 503，直到前端重载；现在只有这个请求回 503。只有 token 流彻底坏掉（两个调度器同时写），前端才会一直报错，直到重载。
- 框架本体的 `ControlLink` 多了一个只读属性 `address`（进程管家的地址），给自己开线程做查找的单元用。

## 还没做

- 版本切换期间让新请求排队，排满了回 503 + Retry-After。
- n > 1、重复惩罚。
- 前端重载时保住进行中的 HTTP 连接：重载会结束这些连接，并发出取消。

## 运行

```sh
uv sync --python 3.12
uv run pytest tests/units/frontend -q
PYTHONPATH=engine:. uv run --extra frontend python -m tests.units.frontend.tokenizer_smoke
PYTHONPATH=engine:. uv run python -m tests.units.frontend.demo
```

demo 打印出地址以后，用这个地址发请求：

```sh
curl -N "$FRONTEND_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"toy","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":true}'
```
