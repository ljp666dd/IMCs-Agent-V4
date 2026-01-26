<template>
  <div class="process_box" v-show="isCollapse">
    <div class="body-box">
      <div class="out_list">
        <div class="out_item" v-for="(item, index) in outList" :key="index">
          <div class="item_text" v-if="index">
            <img class="item_round" :src="item.url" alt="" />
            <p class="p">{{ item.goup }}</p>
            <span class="span">{{ item.content }}</span>
            <div class="dashed_line">
              <img src="../../assets/chart/dashed_line_arrow.png" alt="" />
            </div>
          </div>
          <div class="centent_box" v-else>
            <div class="item_text">
              <img class="item_round" :src="item.url" alt="" />
              <p class="p">{{ item.goup }}</p>
              <span class="span">{{ item.content }}</span>
            </div>
            <div class="sector_box" v-for="(key, num) in centerList" :key="num">
              <div class="one_box" v-if="!num">
                <img src="../../assets/chart/user.png" alt="" />
                <p>
                  <span>{{ key.getter }}</span>
                </p>
                <img src="../../assets/chart/one_arrow.png" alt="" />
              </div>
              <div class="two_box" v-else>
                <div class="top_box">
                  <p class="arrow_box_left">
                    <img
                      src="../../assets/chart/three_arrow.png"
                      v-show="num === 2"
                      alt=""
                    />
                    <img
                      src="../../assets/chart/five_arrow.png"
                      v-show="num === 4"
                      alt=""
                    />
                  </p>
                  <img
                    class="person"
                    src="../../assets/chart/person.png"
                    alt=""
                  />
                  <p class="arrow_box_right">
                    <img
                      src="../../assets/chart/two_arrow.png"
                      v-show="num === 1"
                      alt=""
                    />
                    <img
                      src="../../assets/chart/four_arrow.png"
                      v-show="num === 3"
                      alt=""
                    />
                  </p>
                </div>
                <div class="admin_box">
                  <span>{{ key.getter }}</span>
                </div>
                <div class="bottom_box" v-if="num != 2">
                  <img class="sit" src="../../assets/chart/sit.png" alt="" />
                  <img
                    class="down_arrow"
                    src="../../assets/chart/down_arrow.png"
                    alt=""
                  />
                  <p>
                    <span>{{ key.typeText }}</span>
                  </p>
                  <img
                    class="up_arrow"
                    src="../../assets/chart/up_arrow.png"
                    alt=""
                  />
                  <img
                    class="stand"
                    src="../../assets/chart/stand.png"
                    alt=""
                    v-if="num != 3"
                  />
                </div>
                <div class="bottom_box" v-else>
                  <img
                    class="stand"
                    src="../../assets/chart/stand.png"
                    alt=""
                  />
                  <img
                    class="up_arrow"
                    src="../../assets/chart/up_arrow.png"
                    alt=""
                  />
                  <p>
                    <span>{{ key.typeText }}</span>
                  </p>
                  <img
                    class="down_arrow"
                    src="../../assets/chart/down_arrow.png"
                    alt=""
                  />
                  <img class="sit" src="../../assets/chart/sit.png" alt="" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import oneRound from '../../assets/chart/one_round.png'
import twoRound from '../../assets/chart/two_round.png'
import threeRound from '../../assets/chart/three_round.png'
import fourRound from '../../assets/chart/four_round.png'
import fiveRound from '../../assets/chart/five_round.png'
const isCollapse = ref<boolean>(true)
const outList = ref<Array<any>>([
  {
    url: oneRound,
    goup: 'G1',
    content: 'ORCHESTRATOR'
  },
  {
    url: twoRound,
    goup: 'G2',
    content: 'SCIENTIST'
  },
  {
    url: threeRound,
    goup: 'G3',
    content: 'ENGINEER'
  },
  {
    url: fourRound,
    goup: 'G4',
    content: 'EXECUTOR'
  },
  {
    url: fiveRound,
    goup: 'G5',
    content: 'ANALYST'
  }
])
const centerList = ref<Array<any>>([
  {
    getter: 'Human',
    typeText: ''
  },
  {
    getter: 'Admin',
    typeText: 'Handoff'
  },
  {
    getter: 'Admin',
    typeText: 'Handoff'
  },
  {
    getter: 'Admin',
    typeText: 'Handoff'
  },
  {
    getter: 'Admin',
    typeText: 'Handoff'
  }
])
</script>

<style scoped>
@import './index.less';
</style>
