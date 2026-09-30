        if request is None:
            return False
        self._retire(request, RequestState.CANCELLED, "client_cancelled")
        return True

    def _retire(self, request: Request, state: RequestState, reason: str) -> None:
        """Move a request to a terminal state and make its KV capacity reusable."""
        request.state = state
        request.finish_reason = reason
        self.terminal[request.request_id] = request

    def _reserved_kv_slots(self) -> int:
        """Return conservative cache reservation for all running requests."""
        return sum(item.prompt_tokens + item.max_new_tokens for item in self.running.values())

    def _can_admit(self, request: Request) -> bool:
        """Check active-count and full-lifetime KV capacity before admission."""
        required = request.prompt_tokens + request.max_new_tokens
        return (
            len(self.running) < self.config.max_running_requests
            and self._reserved_kv_slots() + required <= self.config.max_kv_slots
        )

    def _retire_completed(self) -> None:
        """Evict completed requests before forming the next execution plan."""
        for request_id, request in list(self.running.items()):
            if request.output_tokens >= request.max_new_tokens and request.remaining_prompt == 0:
                del self.running[request_id]
                self._retire(request, RequestState.FINISHED, "length")

    def schedule(self) -> List[ScheduledWork]:
        """Build one packed execution plan without doing GPU/model computation.

        Decode gets one token position per active decoding request. Remaining
        capacity advances existing prefills and admits FIFO waiters. All selected
        work must fit the per-iteration token budget.
        """
        self._retire_completed()
        budget = self.config.max_batched_tokens
        plan: List[ScheduledWork] = []

        # Existing decode first: protect time-per-output-token for streaming users.
        for request in self.running.values():
            if request.in_decode and budget:
                plan.append(ScheduledWork(request.request_id, "decode", 1))
                budget -= 1

        # Resume chunked prefills already admitted in earlier iterations.
        for request in self.running.values():
            if request.remaining_prompt and budget:
                count = min(request.remaining_prompt, self.config.prefill_chunk_size, budget)
                plan.append(ScheduledWork(request.request_id, "prefill", count))
                budget -= count

        # Admit new requests only after preserving existing active work.
        while self.waiting and budget and len(self.running) < self.config.max_running_requests:
            candidate = self.waiting[0]
            if not self._can_admit(candidate):
                # FIFO fairness: do not let a too-large request starve indefinitely.
                break
            self.waiting.popleft()
            candidate.state = RequestState.RUNNING
            self.running[candidate.request_id] = candidate
            if candidate.remaining_prompt:
                count = min(candidate.remaining_prompt, self.config.prefill_chunk_size, budget)
                plan.append(ScheduledWork(candidate.request_id, "prefill", count))
                budget -= count
            elif candidate.max_new_tokens == 0:
                self._retire_completed()
            # A zero-length prompt with output work will decode next iteration.
        return plan

    def execute(self, plan: Iterable[ScheduledWork]) -> None:
        """Apply deterministic model results for an execution plan.

        In production, this method is replaced by a model runner: pack tokens,
        positions and KV block tables; launch GPU kernels; sample logits; then
        apply completion/cancellation events. The scheduler only owns state.
        """
        for work in plan:
            request = self.running.get(work.request_id)
            if request is None:
                continue  # Cancellation can race with an already-built plan.
            if work.phase == "prefill":
                request.prompt_processed += work.token_count
                if request.prompt_processed > request.prompt_tokens:
                    self._retire(request, RequestState.FAILED, "prefill_overflow")
                    del self.running[request.request_id]
            elif work.phase == "decode":
                request.output_tokens += 1
                request.generated.append(request.output_tokens)  # stand-in sampled token ID
            else:
                self._retire(request, RequestState.FAILED, "unknown_phase")
                del self.running[request.request_id]
        self._retire_completed()

    def step(self) -> List[ScheduledWork]:
        """Run exactly one admission/execution/eviction iteration and return its plan."""
        plan = self.schedule()
        self.execute(plan)
        self.iteration += 1
        return plan

    def is_idle(self) -> bool:
        """Return True only when no waiting or active requests remain."""
        return not self.waiting and not self.running


def demo() -> None:
    """Demonstrate arrival, prefill, decode, eviction, and replacement admission."""
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(max_running_requests=3, max_batched_tokens=10, max_kv_slots=100, prefill_chunk_size=8)
    )
    arrivals = {
        0: [Request("A", prompt_tokens=2, max_new_tokens=2), Request("B", prompt_tokens=1, max_new_tokens=5)],
        1: [Request("C", prompt_tokens=8, max_new_tokens=3)],
        3: [Request("D", prompt_tokens=10, max_new_tokens=2)],
    }
    while not scheduler.is_idle() or arrivals:
        for request in arrivals.pop(scheduler.iteration, []):
            scheduler.submit(request)
        plan = scheduler.step()
        readable = ", ".join(f"{w.request_id}:{w.phase}[{w.token_count}]" for w in plan) or "idle"
        print(f"iteration={scheduler.iteration - 1:<2} plan={readable:<45} "
              f"running={list(scheduler.running)} waiting={[r.request_id for r in scheduler.waiting]}")
    print("finished:", {key: item.finish_reason for key, item in scheduler.terminal.items()})


