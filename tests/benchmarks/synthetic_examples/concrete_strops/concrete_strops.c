/* concrete_strops: dynamically-linked concrete strcmp/strlen/memcpy+malloc loop.
   Re-creation of the angr-a8epx fixture (original scratchpad copy was wiped).
   Exercises the library-hook dispatch path: every call resolves through the
   PLT into libc, so the hooks are non-main-object and only fire natively when
   prefer_native_library_hooks is on. */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static const char *WORDS[8] = {
    "alpha", "bravo", "charlie", "delta",
    "echo",  "foxtrot", "golf",  "hotel",
};

int main(void) {
    char key[16];
    int hits = 0;

    if (read(0, key, 8) < 0) return 1;
    key[8] = '\0';

    for (int i = 0; i < 8; i++) {
        size_t n = strlen(WORDS[i]);
        char *buf = malloc(n + 1);
        if (!buf) return 1;
        memcpy(buf, WORDS[i], n + 1);
        if (strcmp(buf, "delta") == 0) hits++;
        if (strlen(buf) == 5) hits++;
        free(buf);
    }
    if (hits == 5) { puts("WIN"); return 0; }
    puts("LOSE");
    return 0;
}
