<template>
  <div>
    <SwitchLanguage />
    <router-view></router-view>
  </div>
</template>
<script setup lang="ts">
import SwitchLanguage from './components/SwitchLanguage.vue'
</script>
<style scoped></style>
