<script lang="ts">
  interface Props {
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    disabled?: boolean;
    loading?: boolean;
    fullWidth?: boolean;
    href?: string;
    type?: 'button' | 'submit' | 'reset';
    onclick?: () => void;
    class?: string;
    children?: any;
  }

  let {
    variant = 'primary',
    size = 'md',
    disabled = false,
    loading = false,
    fullWidth = false,
    href,
    type = 'button',
    onclick,
    class: className = '',
    children
  }: Props = $props();
</script>

{#if href}
  <a
    {href}
    class="btn btn-{variant} btn-{size} {className}"
    class:w-full={fullWidth}
    class:opacity-50={disabled}
    class:cursor-not-allowed={disabled}
  >
    {#if loading}
      <div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
    {/if}
    {@render children?.()}
  </a>
{:else}
  <button
    {type}
    class="btn btn-{variant} btn-{size} {className}"
    class:w-full={fullWidth}
    {disabled}
    {onclick}
  >
    {#if loading}
      <div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
    {/if}
    {@render children?.()}
  </button>
{/if}