<script lang="ts">
    interface SelectiveLossConfig {
        selective_loss: boolean;
        selective_loss_level: string;
        selective_loss_schema_keys: string;
        selective_loss_masking_strategy: string;
        selective_loss_masking_start_epoch: number;
        selective_loss_mask_every_n_steps: number;
        selective_loss_mask_for_n_steps: number;
        selective_loss_structural_weight: number;
        selective_loss_verbose: boolean;
    }

    interface Props {
        config: SelectiveLossConfig;
        numEpochs: number;
    }

    let { config = $bindable(), numEpochs }: Props = $props();
</script>

<div>
    <h3 class="text-lg font-semibold text-gray-900 mb-4">
        🎯 Selective Loss (Structured Outputs)
    </h3>

    <div
        class="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg mb-4"
    >
        <p class="text-sm text-gray-800 mb-2">
            <strong>🔬 Experimental Feature:</strong> Optimize training for structured
            outputs (JSON, forms, etc.)
        </p>
        <p class="text-xs text-gray-700">
            Masks structural tokens (braces, colons, whitespace) so the model
            focuses on semantic content. Useful for form extraction, structured
            data generation, and similar tasks.
        </p>
    </div>

    <div class="space-y-4">
        <div>
            <div class="flex items-center">
                <input
                    type="checkbox"
                    id="selective_loss"
                    bind:checked={config.selective_loss}
                    class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label
                    for="selective_loss"
                    class="ml-2 block text-sm font-medium text-gray-700"
                >
                    Enable Selective Loss Masking
                </label>
            </div>
            <p class="text-xs text-gray-500 mt-1 ml-6">
                Automatically mask JSON structural tokens during training
            </p>
        </div>

        {#if config.selective_loss}
            <div
                class="ml-6 space-y-4 p-4 bg-white border border-gray-200 rounded-lg"
            >
                <div>
                    <label
                        for="selective_loss_level"
                        class="block text-sm font-medium text-gray-700 mb-2"
                    >
                        Masking Level
                    </label>
                    <select
                        id="selective_loss_level"
                        bind:value={config.selective_loss_level}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="conservative"
                            >Conservative (Structure Only)</option
                        >
                        <option value="moderate"
                            >Moderate (Structure + null)</option
                        >
                        <option value="aggressive"
                            >Aggressive (Structure + null + Schema Keys)</option
                        >
                    </select>
                    <div class="mt-2 p-3 bg-gray-50 rounded-lg">
                        <p class="text-xs text-gray-700">
                            {#if config.selective_loss_level === "conservative"}
                                <strong>Conservative:</strong> Masks JSON
                                structural characters:
                                <code>{`{, }, [, ], :, ,, "`}</code> and
                                whitespace. Masks ~31% of tokens.
                                <em>Recommended for most cases.</em>
                            {:else if config.selective_loss_level === "moderate"}
                                <strong>Moderate:</strong> Conservative + masks
                                <code>null</code> keyword. Good when null values
                                are predictable.
                            {:else}
                                <strong>Aggressive:</strong> Moderate + masks schema
                                field names. Maximum focus on semantic content. Requires
                                specifying schema keys below.
                            {/if}
                        </p>
                    </div>
                </div>

                {#if config.selective_loss_level === "aggressive"}
                    <div>
                        <label
                            for="selective_loss_schema_keys"
                            class="block text-sm font-medium text-gray-700 mb-1"
                        >
                            Schema Keys to Mask
                        </label>
                        <input
                            type="text"
                            id="selective_loss_schema_keys"
                            bind:value={config.selective_loss_schema_keys}
                            placeholder="Marque,Modele,contents,confidence_score"
                            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        />
                        <p class="text-xs text-gray-500 mt-1">
                            Comma-separated list of JSON field names to mask
                            (e.g., "name,address,phone")
                        </p>
                        <div
                            class="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded"
                        >
                            <p class="text-xs text-yellow-800">
                                ⚠️ Only mask keys that are predictable and don't
                                carry semantic meaning. The model should still
                                learn what values go with each key.
                            </p>
                        </div>
                    </div>
                {/if}

                <div>
                    <label
                        for="selective_loss_masking_strategy"
                        class="block text-sm font-medium text-gray-700 mb-2"
                    >
                        Masking Strategy
                    </label>
                    <select
                        id="selective_loss_masking_strategy"
                        bind:value={config.selective_loss_masking_strategy}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="epoch_based"
                            >Epoch-based (Enable after threshold)</option
                        >
                        <option value="alternating"
                            >Alternating (Cycle ON/OFF)</option
                        >
                        <option value="weighted"
                            >Weighted (Soft per-token weights)</option
                        >
                    </select>
                    <div class="mt-2 p-3 bg-blue-50 rounded-lg">
                        <p class="text-xs text-blue-700">
                            {#if config.selective_loss_masking_strategy === "epoch_based"}
                                <strong>📅 Epoch-based:</strong> Enable masking after
                                a certain epoch. Good for initial experiments and
                                understanding masking impact.
                            {:else if config.selective_loss_masking_strategy === "alternating"}
                                <strong>🔄 Alternating:</strong> Continuously cycle
                                between learning structure and semantics. Recommended
                                for balanced training and avoiding structure degradation.
                            {:else}
                                <strong>⚖️ Weighted:</strong> Soft masking with reduced
                                weight for structural tokens (0.0-1.0). Most flexible
                                approach - structure contributes throughout training
                                but with lower emphasis.
                            {/if}
                        </p>
                    </div>
                </div>

                {#if config.selective_loss_masking_strategy === "epoch_based"}
                    <div>
                        <label
                            for="selective_loss_masking_start_epoch"
                            class="block text-sm font-medium text-gray-700 mb-2"
                        >
                            Masking Start Epoch: {config.selective_loss_masking_start_epoch}
                        </label>
                        <input
                            type="range"
                            id="selective_loss_masking_start_epoch"
                            bind:value={
                                config.selective_loss_masking_start_epoch
                            }
                            min="0"
                            max={numEpochs}
                            step="0.1"
                            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-500"
                        />
                        <div
                            class="flex justify-between text-xs text-gray-500 mt-1"
                        >
                            <span>0.0 (Immediate)</span>
                            {#if numEpochs > 1}
                                <span>{(numEpochs / 2).toFixed(1)}</span>
                            {/if}
                            <span>{numEpochs}.0 epochs</span>
                        </div>
                        <div class="mt-2 p-3 bg-green-50 rounded-lg">
                            <p class="text-xs text-green-700">
                                {#if config.selective_loss_masking_start_epoch === 0.0}
                                    <em>Masking starts immediately</em>
                                {:else}
                                    <em
                                        >Model learns structure for {config.selective_loss_masking_start_epoch}
                                        epochs, then masking begins</em
                                    >
                                {/if}
                            </p>
                        </div>
                    </div>
                {:else if config.selective_loss_masking_strategy === "alternating"}
                    <!-- Alternating Strategy Controls -->
                    <div class="space-y-4">
                        <div>
                            <label
                                for="selective_loss_mask_every_n_steps"
                                class="block text-sm font-medium text-gray-700 mb-2"
                            >
                                Cycle Length (steps): {config.selective_loss_mask_every_n_steps}
                            </label>
                            <input
                                type="range"
                                id="selective_loss_mask_every_n_steps"
                                bind:value={
                                    config.selective_loss_mask_every_n_steps
                                }
                                min="20"
                                max="500"
                                step="10"
                                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                            <div
                                class="flex justify-between text-xs text-gray-500 mt-1"
                            >
                                <span>20 (Frequent)</span>
                                <span>250</span>
                                <span>500 (Long cycles)</span>
                            </div>
                            <p class="text-xs text-gray-600 mt-2">
                                Total steps per cycle (masking ON + masking OFF)
                            </p>
                        </div>

                        <div>
                            <label
                                for="selective_loss_mask_for_n_steps"
                                class="block text-sm font-medium text-gray-700 mb-2"
                            >
                                Masking ON per cycle (steps): {config.selective_loss_mask_for_n_steps}
                            </label>
                            <input
                                type="range"
                                id="selective_loss_mask_for_n_steps"
                                bind:value={
                                    config.selective_loss_mask_for_n_steps
                                }
                                min="10"
                                max={config.selective_loss_mask_every_n_steps}
                                step="5"
                                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
                            />
                            <div
                                class="flex justify-between text-xs text-gray-500 mt-1"
                            >
                                <span>10</span>
                                <span
                                    >{Math.floor(
                                        config.selective_loss_mask_every_n_steps /
                                            2,
                                    )}</span
                                >
                                <span
                                    >{config.selective_loss_mask_every_n_steps}</span
                                >
                            </div>
                            <p class="text-xs text-gray-600 mt-2">
                                Steps with masking ON (rest of cycle has masking
                                OFF)
                            </p>
                        </div>

                        <div
                            class="p-3 bg-purple-50 border border-purple-200 rounded-lg"
                        >
                            <p class="text-xs text-purple-800">
                                <strong>Current pattern:</strong><br />
                                🟢 Steps 0-{config.selective_loss_mask_for_n_steps -
                                    1}: Masking ON (learn semantics)<br />
                                🔴 Steps {config.selective_loss_mask_for_n_steps}-{config.selective_loss_mask_every_n_steps -
                                    1}: Masking OFF (learn structure)<br />
                                Then cycle repeats... ({Math.round(
                                    (config.selective_loss_mask_for_n_steps /
                                        config.selective_loss_mask_every_n_steps) *
                                        100,
                                )}% masking / {100 -
                                    Math.round(
                                        (config.selective_loss_mask_for_n_steps /
                                            config.selective_loss_mask_every_n_steps) *
                                            100,
                                    )}% structure)
                            </p>
                        </div>
                    </div>
                {:else if config.selective_loss_masking_strategy === "weighted"}
                    <!-- Weighted Strategy Controls -->
                    <div class="space-y-4">
                        <div>
                            <label
                                for="selective_loss_structural_weight"
                                class="block text-sm font-medium text-gray-700 mb-2"
                            >
                                Structural Token Weight: {config.selective_loss_structural_weight.toFixed(
                                    2,
                                )}
                            </label>
                            <input
                                type="range"
                                id="selective_loss_structural_weight"
                                bind:value={
                                    config.selective_loss_structural_weight
                                }
                                min="0.0"
                                max="1.0"
                                step="0.05"
                                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                            />
                            <div
                                class="flex justify-between text-xs text-gray-500 mt-1"
                            >
                                <span>0.0 (Ignore)</span>
                                <span>0.5 (Balanced)</span>
                                <span>1.0 (Full weight)</span>
                            </div>
                            <p class="text-xs text-gray-600 mt-2">
                                Weight applied to structural tokens during loss
                                calculation. Lower = less emphasis on structure,
                                higher = more emphasis.
                            </p>
                        </div>

                        <div
                            class="p-3 bg-amber-50 border border-amber-200 rounded-lg"
                        >
                            <p class="text-xs text-amber-800">
                                <strong>Current weighting:</strong><br />
                                🔧 Structural tokens (JSON syntax, keys):
                                <strong
                                    >{config.selective_loss_structural_weight.toFixed(
                                        2,
                                    )}×</strong
                                >
                                weight<br />
                                📝 Semantic tokens (values, content):
                                <strong>1.00×</strong>
                                weight<br />
                                <br />
                                {#if config.selective_loss_structural_weight < 0.1}
                                    <em
                                        >Very low structure emphasis - model may
                                        struggle with formatting</em
                                    >
                                {:else if config.selective_loss_structural_weight < 0.3}
                                    <em
                                        >Recommended for structured outputs -
                                        good balance</em
                                    >
                                {:else if config.selective_loss_structural_weight < 0.7}
                                    <em
                                        >Moderate structure emphasis - more
                                        balanced training</em
                                    >
                                {:else}
                                    <em
                                        >High structure emphasis - close to
                                        unweighted training</em
                                    >
                                {/if}
                            </p>
                        </div>

                        <div
                            class="p-3 bg-blue-50 border border-blue-200 rounded-lg"
                        >
                            <p class="text-xs text-blue-700">
                                <strong>💡 Tip:</strong> Start with 0.10 (default)
                                and adjust based on results. Lower values (0.05-0.15)
                                work well for structured outputs where semantic content
                                is most important. Higher values (0.3-0.5) provide
                                more balanced training.
                            </p>
                        </div>
                    </div>
                {/if}

                <div>
                    <div class="flex items-center">
                        <input
                            type="checkbox"
                            id="selective_loss_verbose"
                            bind:checked={config.selective_loss_verbose}
                            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                            for="selective_loss_verbose"
                            class="ml-2 block text-sm text-gray-700"
                        >
                            Verbose mode (print masking statistics)
                        </label>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">
                        Display detailed token masking stats during training
                    </p>
                </div>

                <div class="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <h4 class="text-sm font-semibold text-blue-900 mb-2">
                        📊 What Gets Masked?
                    </h4>
                    <ul class="text-xs text-blue-800 space-y-1">
                        <li>
                            ✓ Structural: <code>{`{ } [ ] : ,`}</code> and whitespace
                            (spaces, newlines, tabs)
                        </li>
                        <li>
                            ✓ Quotes: <code>"</code> (string delimiters - purely
                            structural)
                        </li>
                        <li>
                            ✓ Null keyword: <code>null</code> (moderate/aggressive
                            only)
                        </li>
                        <li>
                            ✗ NOT masked: <code>true</code>, <code>false</code> (can
                            be semantic)
                        </li>
                        <li>
                            ✓ Schema keys: Field names like <code>name</code> (aggressive
                            only)
                        </li>
                    </ul>
                    <p class="text-xs text-blue-700 mt-2">
                        <strong>Example:</strong> In
                        <code>{`{"name": "John", "age": 30}`}</code>,
                        conservative mode masks <code>{`{ } : , "`}</code> and
                        spaces (~31% of tokens), trains on
                        <code>name John age 30</code>
                    </p>
                </div>
            </div>
        {/if}
    </div>
</div>
