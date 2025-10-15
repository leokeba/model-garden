<script lang="ts">
  interface Props {
    variant?: 'success' | 'warning' | 'error' | 'info';
    size?: 'sm' | 'md';
    class?: string;
    children?: any;
  }

  let {
    variant = 'info',
    size = 'md',
    class: className = '',
    children
  }: Props = $props();
</script>

<span class="badge badge-{variant} badge-{size} {className}">
  {@render children?.()}
</span>