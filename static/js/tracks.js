/* Reading-track toggle + quiz reveal wiring.
 *
 * Track state lives in localStorage['probintro-track'] ('all' | 'beginner' |
 * 'cogsci') and is mirrored onto <html data-track="..."> (set pre-paint by
 * the inline script in custom-header.html). CSS does the actual dimming and
 * the next-in-track footer selection; this file only moves state around.
 */
(function () {
  'use strict';

  var KEY = 'probintro-track';

  function currentTrack() {
    return document.documentElement.getAttribute('data-track') || 'all';
  }

  function setTrack(track) {
    try { localStorage.setItem(KEY, track); } catch (e) { /* private mode */ }
    document.documentElement.setAttribute('data-track', track);
    updateButtons();
  }

  function updateButtons() {
    var track = currentTrack();
    document.querySelectorAll('[data-track-choice]').forEach(function (btn) {
      var active = btn.getAttribute('data-track-choice') === track;
      btn.classList.toggle('active', active);
      if (active) { btn.setAttribute('aria-pressed', 'true'); }
      else { btn.removeAttribute('aria-pressed'); }
    });
  }

  function liftTrackTags() {
    // menu-post.html renders <span class="track-tags" data-tracks="..."> inside
    // each sidebar link; copy the attribute onto the enclosing <li> so CSS can
    // dim whole menu rows without depending on :has().
    document.querySelectorAll('#R-sidebar .track-tags[data-tracks]').forEach(function (tag) {
      var li = tag.closest('li');
      if (li && !li.hasAttribute('data-tracks')) {
        li.setAttribute('data-tracks', tag.getAttribute('data-tracks'));
      }
    });
  }

  function wireQuizzes() {
    document.querySelectorAll('form.pi-quiz').forEach(function (form) {
      var check = form.querySelector('.pi-quiz-check');
      if (!check) { return; }
      check.addEventListener('click', function () {
        form.classList.add('revealed');
        check.disabled = true;
      });
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    liftTrackTags();
    updateButtons();
    wireQuizzes();
    document.addEventListener('click', function (ev) {
      var btn = ev.target.closest('[data-track-choice]');
      if (btn) { setTrack(btn.getAttribute('data-track-choice')); }
    });
  });
}());
